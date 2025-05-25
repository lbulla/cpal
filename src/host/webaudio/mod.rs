extern crate js_sys;
extern crate wasm_bindgen;
extern crate web_sys;

use self::js_sys::{global, Array, Float32Array, Reflect, SharedArrayBuffer};
use self::wasm_bindgen::prelude::*;
use self::wasm_bindgen::JsCast;
use self::web_sys::{
    window, AudioContext, AudioContextOptions, AudioWorkletNode, AudioWorkletNodeOptions, Blob,
    BlobPropertyBag, MediaDevices, MediaStream, MediaStreamAudioSourceNode, MediaStreamConstraints,
    MediaTrackConstraints, MessageEvent, Url,
};
use crate::traits::{DeviceTrait, HostTrait, StreamTrait};
use crate::{
    BackendSpecificError, BufferSize, BuildStreamError, Data, DefaultStreamConfigError,
    DeviceNameError, DevicesError, InputCallbackInfo, InputStreamTimestamp, OutputCallbackInfo,
    OutputStreamTimestamp, PauseStreamError, PlayStreamError, SampleFormat, SampleRate,
    StreamConfig, StreamError, StreamInstant, SupportedBufferSize, SupportedStreamConfig,
    SupportedStreamConfigRange, SupportedStreamConfigsError,
};
use std::cell::RefCell;
use std::rc::Rc;
use std::time::Duration;
use wasm_bindgen_futures::{spawn_local, JsFuture};

/// Content is false if the iterator is empty.
pub struct Devices(bool);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Device;

pub struct Host;

pub struct Stream {
    inner: Rc<StreamInner>,
}

struct StreamInner {
    ctx: Rc<AudioContext>,
    stream_type: RefCell<Option<StreamType>>,
}

struct InputStream {
    node: AudioWorkletNode,
    _source: MediaStreamAudioSourceNode,
    _on_process_closure: Closure<dyn FnMut(MessageEvent)>,
}

impl InputStream {
    // Fixed, see: https://developer.mozilla.org/en-US/docs/Web/API/MediaStreamAudioSourceNode.
    const NUM_CHANNELS: u16 = 2;

    const PROCESSOR_NAME: &'static str = "web-input-processor";
    const PROCESSOR_CODE: &'static str = r#"
class WebInputProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super(options);
        this.buffer = options.processorOptions.buffer;
        this.samplesWritten = 0;
    }

    process(inputs, outputs, parameters) {
        for (let s = 0; s < inputs[0][0].length; s++) {
            for (let c = 0; c < inputs[0].length; c++) {
                this.buffer[this.samplesWritten] = inputs[0][c][s];
                this.samplesWritten += 1;
            }
            if (this.samplesWritten >= this.buffer.length) {
                this.port.postMessage(0);
                this.samplesWritten = 0;
            }
        }
        return true;
    }
}

registerProcessor('web-input-processor', WebInputProcessor);
"#;
}

struct OutputStream {
    node: AudioWorkletNode,
    _on_process_closure: Closure<dyn FnMut(MessageEvent)>,
}

impl OutputStream {
    const MIN_CHANNELS: u16 = 1;
    const MAX_CHANNELS: u16 = 32;

    const PROCESSOR_NAME: &'static str = "web-output-processor";
    const PROCESSOR_CODE: &'static str = r#"
class WebOutputProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super(options);
        this.buffer = options.processorOptions.buffer;
        this.samplesRead = 0;
        this.port.postMessage(0);
    }

    process(inputs, outputs, parameters) {
        for (let s = 0; s < outputs[0][0].length; s++) {
            for (let c = 0; c < outputs[0].length; c++) {
                outputs[0][c][s] = this.buffer[this.samplesRead];
                this.samplesRead += 1;
                if (this.samplesRead >= this.buffer.length) {
                    this.port.postMessage(0);
                    this.samplesRead = 0;
                }
            }
        }
        return true;
    }
}

registerProcessor('web-output-processor', WebOutputProcessor);
"#;
}

// Only used as storage to handle WebAudio's async nature.
#[allow(unused)]
enum StreamType {
    Input(InputStream),
    Output(OutputStream),
}

struct BufferSizes {
    frames: usize,
    samples: usize,
    secs: f64,
}

struct ClosureParams {
    buffer: Float32Array,
    temp_buffer: Vec<f32>,
    js_buffer_factor: usize,
    buffer_size_secs: f64,
    time_at_start_of_buffer: f64,
}

pub type SupportedInputConfigs = ::std::vec::IntoIter<SupportedStreamConfigRange>;
pub type SupportedOutputConfigs = ::std::vec::IntoIter<SupportedStreamConfigRange>;

const MIN_SAMPLE_RATE: SampleRate = SampleRate(8_000);
const MAX_SAMPLE_RATE: SampleRate = SampleRate(96_000);
const DEFAULT_SAMPLE_RATE: SampleRate = SampleRate(44_100);
const MIN_BUFFER_SIZE: u32 = 1;
const MAX_BUFFER_SIZE: u32 = u32::MAX;
const DEFAULT_BUFFER_SIZE: usize = 2048;
// The buffer size processors are currently limited to 128.
// See: https://developer.mozilla.org/en-US/docs/Web/API/AudioWorkletProcessor/process.
const WORKLET_BUFFER_SIZE: usize = 128;
const SUPPORTED_SAMPLE_FORMAT: SampleFormat = SampleFormat::F32;

impl Host {
    pub fn new() -> Result<Self, crate::HostUnavailable> {
        Ok(Host)
    }
}

impl HostTrait for Host {
    type Devices = Devices;
    type Device = Device;

    fn is_available() -> bool {
        // Assume this host is always available on WebAudio.
        true
    }

    fn devices(&self) -> Result<Self::Devices, DevicesError> {
        Devices::new()
    }

    fn default_input_device(&self) -> Option<Self::Device> {
        default_input_device()
    }

    fn default_output_device(&self) -> Option<Self::Device> {
        default_output_device()
    }
}

impl Devices {
    fn new() -> Result<Self, DevicesError> {
        Ok(Self::default())
    }
}

impl Device {
    #[inline]
    fn name(&self) -> Result<String, DeviceNameError> {
        Ok("Default Device".to_owned())
    }

    #[inline]
    fn supported_input_configs(
        &self,
    ) -> Result<SupportedInputConfigs, SupportedStreamConfigsError> {
        let configs = vec![SupportedStreamConfigRange::new(
            InputStream::NUM_CHANNELS,
            MIN_SAMPLE_RATE,
            MAX_SAMPLE_RATE,
            SupportedBufferSize::Range {
                min: MIN_BUFFER_SIZE,
                max: MAX_BUFFER_SIZE,
            },
            SUPPORTED_SAMPLE_FORMAT,
        )];
        Ok(configs.into_iter())
    }

    #[inline]
    fn supported_output_configs(
        &self,
    ) -> Result<SupportedOutputConfigs, SupportedStreamConfigsError> {
        let buffer_size = SupportedBufferSize::Range {
            min: MIN_BUFFER_SIZE,
            max: MAX_BUFFER_SIZE,
        };
        let configs: Vec<_> = (OutputStream::MIN_CHANNELS..=OutputStream::MAX_CHANNELS)
            .map(|channels| SupportedStreamConfigRange {
                channels,
                min_sample_rate: MIN_SAMPLE_RATE,
                max_sample_rate: MAX_SAMPLE_RATE,
                buffer_size: buffer_size.clone(),
                sample_format: SUPPORTED_SAMPLE_FORMAT,
            })
            .collect();
        Ok(configs.into_iter())
    }

    #[inline]
    fn default_input_config(&self) -> Result<SupportedStreamConfig, DefaultStreamConfigError> {
        const EXPECT: &str = "expected at least one valid WebAudio stream config";
        let config = self
            .supported_input_configs()
            .expect(EXPECT)
            .max_by(|a, b| a.cmp_default_heuristics(b))
            .unwrap()
            .with_sample_rate(DEFAULT_SAMPLE_RATE);

        Ok(config)
    }

    #[inline]
    fn default_output_config(&self) -> Result<SupportedStreamConfig, DefaultStreamConfigError> {
        const EXPECT: &str = "expected at least one valid WebAudio stream config";
        let config = self
            .supported_output_configs()
            .expect(EXPECT)
            .max_by(|a, b| a.cmp_default_heuristics(b))
            .unwrap()
            .with_sample_rate(DEFAULT_SAMPLE_RATE);

        Ok(config)
    }

    fn buffer_sizes(config: &StreamConfig) -> Result<BufferSizes, BuildStreamError> {
        let frames = match config.buffer_size {
            BufferSize::Fixed(v) => {
                if v == 0 {
                    return Err(BuildStreamError::StreamConfigNotSupported);
                } else {
                    v as usize
                }
            }
            BufferSize::Default => DEFAULT_BUFFER_SIZE,
        };
        let samples = frames * config.channels as usize;
        let secs = buffer_time_step_secs(frames, config.sample_rate);

        Ok(BufferSizes {
            frames,
            samples,
            secs,
        })
    }

    fn create_inner(config: &StreamConfig) -> Result<Rc<StreamInner>, JsValue> {
        let stream_opts = AudioContextOptions::new();
        stream_opts
            .set_latency_hint(&(WORKLET_BUFFER_SIZE as f32 / config.sample_rate.0 as f32).into());
        stream_opts.set_sample_rate(config.sample_rate.0 as _);
        let ctx = AudioContext::new_with_context_options(&stream_opts)?;

        Ok(Rc::new(StreamInner {
            ctx: Rc::new(ctx),
            stream_type: RefCell::new(None),
        }))
    }

    async fn create_node(
        ctx: &AudioContext,
        buffer_sizes: &BufferSizes,
        n_channels: usize,
        processor_name: &'static str,
        processor_code: &'static str,
    ) -> Result<(AudioWorkletNode, ClosureParams), JsValue> {
        let blob_parts = Array::new();
        blob_parts.push(&processor_code.into());
        let blob_parts = JsValue::from(blob_parts);
        let blob_options = BlobPropertyBag::new();
        blob_options.set_type("application/javascript");
        let blob = Blob::new_with_str_sequence_and_options(&blob_parts, &blob_options)?;
        let url = Url::create_object_url_with_blob(&blob)?;

        let module = ctx.audio_worklet()?.add_module(&url)?;
        JsFuture::from(module).await?;
        Url::revoke_object_url(&url)?;

        let mut buffer_size_js = buffer_sizes.frames.max(WORKLET_BUFFER_SIZE) * n_channels;
        let js_buffer_factor =
            (buffer_size_js as f32 / buffer_sizes.samples as f32).ceil() as usize;
        if js_buffer_factor != 1 {
            buffer_size_js = buffer_sizes.samples * js_buffer_factor;
        }
        let buffer = Float32Array::new(&SharedArrayBuffer::new(
            (buffer_size_js * size_of::<f32>()) as _,
        ));

        let options = AudioWorkletNodeOptions::new();
        let object = js_sys::Object::new();
        Reflect::set(&object, &"buffer".into(), &buffer)?;
        options.set_processor_options(Some(&object));
        let node = AudioWorkletNode::new_with_options(ctx, processor_name, &options)?;

        let temp_buffer = vec![0.0; buffer_sizes.samples];
        let time_at_start_of_buffer = 0.0;

        Ok((
            node,
            ClosureParams {
                buffer,
                temp_buffer,
                js_buffer_factor,
                buffer_size_secs: buffer_sizes.secs,
                time_at_start_of_buffer,
            },
        ))
    }
}

impl DeviceTrait for Device {
    type SupportedInputConfigs = SupportedInputConfigs;
    type SupportedOutputConfigs = SupportedOutputConfigs;
    type Stream = Stream;

    #[inline]
    fn name(&self) -> Result<String, DeviceNameError> {
        Device::name(self)
    }

    #[inline]
    fn supported_input_configs(
        &self,
    ) -> Result<Self::SupportedInputConfigs, SupportedStreamConfigsError> {
        Device::supported_input_configs(self)
    }

    #[inline]
    fn supported_output_configs(
        &self,
    ) -> Result<Self::SupportedOutputConfigs, SupportedStreamConfigsError> {
        Device::supported_output_configs(self)
    }

    #[inline]
    fn default_input_config(&self) -> Result<SupportedStreamConfig, DefaultStreamConfigError> {
        Device::default_input_config(self)
    }

    #[inline]
    fn default_output_config(&self) -> Result<SupportedStreamConfig, DefaultStreamConfigError> {
        Device::default_output_config(self)
    }

    fn build_input_stream_raw<D, E>(
        &self,
        config: &StreamConfig,
        sample_format: SampleFormat,
        mut data_callback: D,
        mut error_callback: E,
        _timeout: Option<Duration>,
    ) -> Result<Self::Stream, BuildStreamError>
    where
        D: FnMut(&Data, &InputCallbackInfo) + Send + 'static,
        E: FnMut(StreamError) + Send + 'static,
    {
        if !valid_input_config(config, sample_format) {
            return Err(BuildStreamError::StreamConfigNotSupported);
        }

        let n_channels = config.channels as usize;
        let buffer_sizes = Self::buffer_sizes(config)?;
        let inner = Self::create_inner(config).map_err(map_err::<BuildStreamError>)?;

        spawn_local({
            let inner = inner.clone();
            async move {
                let input_stream = async {
                    let devices = devices()?;

                    let web_stream_constraints = MediaStreamConstraints::new();
                    let track_constraints = MediaTrackConstraints::new();
                    // TODO: Settings for these?
                    track_constraints.set_auto_gain_control(&false.into());
                    track_constraints.set_echo_cancellation(&false.into());
                    track_constraints.set_noise_suppression(&false.into());
                    web_stream_constraints.set_audio(&track_constraints.into());

                    let web_stream =
                        devices.get_user_media_with_constraints(&web_stream_constraints)?;
                    let web_stream = JsFuture::from(web_stream);
                    let web_stream = web_stream.await?;
                    let web_stream = web_stream.unchecked_ref::<MediaStream>();
                    let _source = inner.ctx.create_media_stream_source(&web_stream)?;

                    let (node, mut params) = Self::create_node(
                        &inner.ctx,
                        &buffer_sizes,
                        n_channels,
                        InputStream::PROCESSOR_NAME,
                        InputStream::PROCESSOR_CODE,
                    )
                    .await?;

                    let ctx = inner.ctx.clone();
                    let _on_process_closure = if params.js_buffer_factor == 1 {
                        Closure::new(move |_: MessageEvent| {
                            let now = ctx.current_time();

                            params.buffer.copy_to(&mut params.temp_buffer);

                            let data = unsafe {
                                Data::from_parts(
                                    params.temp_buffer.as_mut_ptr() as _,
                                    params.temp_buffer.len(),
                                    sample_format,
                                )
                            };
                            let callback = StreamInstant::from_secs_f64(now);
                            let capture =
                                StreamInstant::from_secs_f64(params.time_at_start_of_buffer);
                            let timestamp = InputStreamTimestamp { callback, capture };
                            let info = InputCallbackInfo::new(timestamp);
                            data_callback(&data, &info);

                            params.time_at_start_of_buffer += params.buffer_size_secs;
                        })
                    } else {
                        Closure::new(move |_: MessageEvent| {
                            let mut now = ctx.current_time();

                            let data = unsafe {
                                Data::from_parts(
                                    params.temp_buffer.as_mut_ptr() as _,
                                    params.temp_buffer.len(),
                                    sample_format,
                                )
                            };

                            for i in 0..params.js_buffer_factor {
                                params
                                    .buffer
                                    .subarray(
                                        (i * params.temp_buffer.len()) as _,
                                        ((i + 1) * params.temp_buffer.len()) as _,
                                    )
                                    .copy_to(&mut params.temp_buffer);

                                let callback = StreamInstant::from_secs_f64(now);
                                let capture =
                                    StreamInstant::from_secs_f64(params.time_at_start_of_buffer);
                                let timestamp = InputStreamTimestamp { callback, capture };
                                let info = InputCallbackInfo::new(timestamp);
                                data_callback(&data, &info);

                                now += params.buffer_size_secs;
                                params.time_at_start_of_buffer += params.buffer_size_secs;
                            }
                        })
                    };
                    node.port()?
                        .set_onmessage(Some(_on_process_closure.as_ref().unchecked_ref()));
                    _source.connect_with_audio_node(&node)?;

                    Ok(InputStream {
                        node,
                        _source,
                        _on_process_closure,
                    })
                };

                match input_stream.await {
                    Ok(s) => {
                        inner.stream_type.replace(Some(StreamType::Input(s)));
                    }
                    Err(err) => {
                        error_callback(map_err(err));
                    }
                }
            }
        });

        Ok(Stream { inner })
    }

    /// Create an output stream.
    fn build_output_stream_raw<D, E>(
        &self,
        config: &StreamConfig,
        sample_format: SampleFormat,
        mut data_callback: D,
        mut error_callback: E,
        _timeout: Option<Duration>,
    ) -> Result<Self::Stream, BuildStreamError>
    where
        D: FnMut(&mut Data, &OutputCallbackInfo) + Send + 'static,
        E: FnMut(StreamError) + Send + 'static,
    {
        if !valid_output_config(config, sample_format) {
            return Err(BuildStreamError::StreamConfigNotSupported);
        }

        let n_channels = config.channels as u32;
        let buffer_sizes = Self::buffer_sizes(config)?;
        let inner = Self::create_inner(config).map_err(map_err::<BuildStreamError>)?;

        spawn_local({
            let inner = inner.clone();
            async move {
                let output_stream = async {
                    let (node, mut params) = Self::create_node(
                        &inner.ctx,
                        &buffer_sizes,
                        n_channels as _,
                        OutputStream::PROCESSOR_NAME,
                        OutputStream::PROCESSOR_CODE,
                    )
                    .await?;

                    node.set_channel_count_mode(web_sys::ChannelCountMode::Explicit);
                    node.set_channel_count(n_channels);

                    let destination = inner.ctx.destination();
                    // If possible, set the destination's channel_count to the given config.channel.
                    // If not, fallback on the default destination channel_count to keep previous
                    // behavior and do not return an error.
                    if n_channels <= destination.max_channel_count() {
                        destination.set_channel_count(n_channels);
                    }

                    let ctx = inner.ctx.clone();
                    let _on_process_closure = if params.js_buffer_factor == 1 {
                        Closure::new(move |_: MessageEvent| {
                            let now = ctx.current_time();

                            let mut data = unsafe {
                                Data::from_parts(
                                    params.temp_buffer.as_mut_ptr() as _,
                                    params.temp_buffer.len(),
                                    sample_format,
                                )
                            };
                            let callback = StreamInstant::from_secs_f64(now);
                            let playback =
                                StreamInstant::from_secs_f64(params.time_at_start_of_buffer);
                            let timestamp = OutputStreamTimestamp { callback, playback };
                            let info = OutputCallbackInfo { timestamp };
                            (data_callback)(&mut data, &info);

                            params.buffer.copy_from(&params.temp_buffer);
                            params.time_at_start_of_buffer += params.buffer_size_secs;
                        })
                    } else {
                        Closure::new(move |_: MessageEvent| {
                            let mut now = ctx.current_time();

                            let mut data = unsafe {
                                Data::from_parts(
                                    params.temp_buffer.as_mut_ptr() as _,
                                    params.temp_buffer.len(),
                                    sample_format,
                                )
                            };

                            for i in 0..params.js_buffer_factor {
                                let callback = StreamInstant::from_secs_f64(now);
                                let playback =
                                    StreamInstant::from_secs_f64(params.time_at_start_of_buffer);
                                let timestamp = OutputStreamTimestamp { callback, playback };
                                let info = OutputCallbackInfo { timestamp };
                                (data_callback)(&mut data, &info);

                                params
                                    .buffer
                                    .subarray(
                                        (i * params.temp_buffer.len()) as _,
                                        ((i + 1) * params.temp_buffer.len()) as _,
                                    )
                                    .copy_from(&params.temp_buffer);

                                now += params.buffer_size_secs;
                                params.time_at_start_of_buffer += params.buffer_size_secs;
                            }
                        })
                    };
                    node.port()?
                        .set_onmessage(Some(_on_process_closure.as_ref().unchecked_ref()));
                    node.connect_with_audio_node(&destination)?;

                    Ok(OutputStream {
                        node,
                        _on_process_closure,
                    })
                };

                match output_stream.await {
                    Ok(s) => {
                        inner.stream_type.replace(Some(StreamType::Output(s)));
                    }
                    Err(err) => {
                        error_callback(map_err(err));
                    }
                }
            }
        });

        Ok(Stream { inner })
    }
}

impl Stream {
    /// Return the [`AudioContext`](https://developer.mozilla.org/docs/Web/API/AudioContext) used
    /// by this stream.
    pub fn audio_context(&self) -> &AudioContext {
        &*self.inner.ctx
    }
}

impl StreamTrait for Stream {
    fn play(&self) -> Result<(), PlayStreamError> {
        match self.inner.ctx.resume() {
            Ok(_) => Ok(()),
            Err(err) => Err(map_err(err)),
        }
    }

    fn pause(&self) -> Result<(), PauseStreamError> {
        match self.inner.ctx.suspend() {
            Ok(_) => Ok(()),
            Err(err) => Err(map_err(err)),
        }
    }
}

impl Drop for StreamInner {
    fn drop(&mut self) {
        let _ = self.ctx.close();
    }
}

impl Drop for InputStream {
    fn drop(&mut self) {
        if let Ok(port) = self.node.port() {
            port.set_onmessage(None);
        }
    }
}

impl Drop for OutputStream {
    fn drop(&mut self) {
        if let Ok(port) = self.node.port() {
            port.set_onmessage(None);
        }
    }
}

impl Default for Devices {
    fn default() -> Devices {
        // We produce an empty iterator if the WebAudio API isn't available.
        Devices(is_webaudio_available())
    }
}

impl Iterator for Devices {
    type Item = Device;
    #[inline]
    fn next(&mut self) -> Option<Device> {
        if self.0 {
            self.0 = false;
            Some(Device)
        } else {
            None
        }
    }
}

fn devices() -> Result<MediaDevices, JsValue> {
    window()
        .map(|w| w.navigator().media_devices())
        .unwrap_or(Err("No devices available".into()))
}

#[inline]
fn default_input_device() -> Option<Device> {
    if is_webaudio_available() {
        Some(Device)
    } else {
        None
    }
}

#[inline]
fn default_output_device() -> Option<Device> {
    if is_webaudio_available() {
        Some(Device)
    } else {
        None
    }
}

// Detects whether the `AudioContext` global variable is available.
fn is_webaudio_available() -> bool {
    Reflect::get(&global(), &JsValue::from("AudioContext"))
        .unwrap()
        .is_truthy()
}

fn map_err<T: From<BackendSpecificError>>(err: JsValue) -> T {
    let description = format!("{:?}", err);
    let err = BackendSpecificError { description };
    err.into()
}

// Whether or not the given stream configuration is valid for building a stream.
fn valid_input_config(conf: &StreamConfig, sample_format: SampleFormat) -> bool {
    conf.channels == InputStream::NUM_CHANNELS
        && conf.sample_rate <= MAX_SAMPLE_RATE
        && conf.sample_rate >= MIN_SAMPLE_RATE
        && sample_format == SUPPORTED_SAMPLE_FORMAT
}

fn valid_output_config(conf: &StreamConfig, sample_format: SampleFormat) -> bool {
    conf.channels <= OutputStream::MAX_CHANNELS
        && conf.channels >= OutputStream::MIN_CHANNELS
        && conf.sample_rate <= MAX_SAMPLE_RATE
        && conf.sample_rate >= MIN_SAMPLE_RATE
        && sample_format == SUPPORTED_SAMPLE_FORMAT
}

fn buffer_time_step_secs(buffer_size_frames: usize, sample_rate: SampleRate) -> f64 {
    buffer_size_frames as f64 / sample_rate.0 as f64
}
