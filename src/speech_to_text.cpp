#include "speech_to_text.h"
#include <libsamplerate/src/samplerate.h>
#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <godot_cpp/classes/audio_server.hpp>
#include <godot_cpp/classes/file_access.hpp>
#include <godot_cpp/classes/project_settings.hpp>
#include <godot_cpp/classes/time.hpp>
#include <godot_cpp/core/error_macros.hpp>
#include <godot_cpp/core/memory.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>
#include <godot_cpp/variant/packed_vector2_array.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <string>
#include <utility>
#include <vector>

namespace {

struct GodotModelLoaderContext {
	Ref<FileAccess> file;
};

bool _open_model_loader(const String &p_path, GodotModelLoaderContext &r_context, whisper_model_loader &r_loader) {
	if (p_path.is_empty()) {
		ERR_PRINT("Whisper model file path is empty.");
		return false;
	}
	if (!FileAccess::file_exists(p_path)) {
		ERR_PRINT("Whisper model file not found: " + p_path);
		return false;
	}
	r_context.file = FileAccess::open(p_path, FileAccess::READ);
	if (r_context.file.is_null() || !r_context.file->is_open()) {
		ERR_PRINT("Whisper model file could not be opened: " + p_path);
		return false;
	}

	r_loader.context = &r_context;
	r_loader.read = [](void *ctx, void *output, size_t read_size) {
		GodotModelLoaderContext *loader_context = reinterpret_cast<GodotModelLoaderContext *>(ctx);
		return (size_t)loader_context->file->get_buffer(reinterpret_cast<uint8_t *>(output), read_size);
	};
	r_loader.eof = [](void *ctx) {
		GodotModelLoaderContext *loader_context = reinterpret_cast<GodotModelLoaderContext *>(ctx);
		return loader_context->file->eof_reached();
	};
	r_loader.close = [](void *ctx) {
		GodotModelLoaderContext *loader_context = reinterpret_cast<GodotModelLoaderContext *>(ctx);
		if (loader_context->file.is_valid()) {
			loader_context->file->close();
		}
	};
	return true;
}

uint32_t _read_le_u32(const uint8_t *p_bytes) {
	return (uint32_t)p_bytes[0] |
			((uint32_t)p_bytes[1] << 8) |
			((uint32_t)p_bytes[2] << 16) |
			((uint32_t)p_bytes[3] << 24);
}

bool _read_exact(const Ref<FileAccess> &p_file, uint8_t *p_dst, uint64_t p_size) {
	return p_file.is_valid() && p_file->get_buffer(p_dst, p_size) == p_size;
}

bool _is_valid_silero_vad_model_file(const String &p_path) {
	Ref<FileAccess> file = FileAccess::open(p_path, FileAccess::READ);
	if (file.is_null() || !file->is_open()) {
		ERR_PRINT("VAD model file could not be opened: " + p_path);
		return false;
	}

	uint8_t data[4];
	if (!_read_exact(file, data, 4) || _read_le_u32(data) != 0x67676d6c) {
		ERR_PRINT("VAD model has invalid ggml magic: " + p_path);
		return false;
	}

	if (!_read_exact(file, data, 4)) {
		ERR_PRINT("VAD model header is incomplete: " + p_path);
		return false;
	}
	const uint32_t model_type_len = _read_le_u32(data);
	if (model_type_len == 0 || model_type_len > 64) {
		ERR_PRINT("VAD model is not a Silero VAD model: " + p_path);
		return false;
	}

	PackedByteArray model_type;
	model_type.resize(model_type_len);
	if (!_read_exact(file, model_type.ptrw(), model_type_len)) {
		ERR_PRINT("VAD model type header is incomplete: " + p_path);
		return false;
	}
	if (model_type_len < 6 || std::memcmp(model_type.ptr(), "silero", 6) != 0) {
		ERR_PRINT("VAD model is not a Silero VAD model: " + p_path);
		return false;
	}

	return true;
}

int _centiseconds_to_samples(float p_centiseconds) {
	return (int)((p_centiseconds / 100.0f) * WHISPER_SAMPLE_RATE + 0.5f);
}

PackedByteArray _bytes_from_c_string(const char *p_text) {
	PackedByteArray bytes;
	if (!p_text) {
		return bytes;
	}
	const size_t len = strlen(p_text);
	bytes.resize(len);
	if (len > 0 && (size_t)bytes.size() >= len) {
		std::memcpy(bytes.ptrw(), p_text, len);
	}
	return bytes;
}

} // namespace

uint32_t _resample_audio_buffer(
		const float *p_src, const uint32_t p_src_frame_count,
		const uint32_t p_src_samplerate, const uint32_t p_target_samplerate,
		float *p_dst,
		SpeechToText::InterpolatorType interpolator_type) {
	if (p_src_samplerate != p_target_samplerate) {
		SRC_DATA src_data;

		src_data.data_in = p_src;
		src_data.data_out = p_dst;

		src_data.input_frames = p_src_frame_count;
		src_data.src_ratio = (double)p_target_samplerate / (double)p_src_samplerate;
		src_data.output_frames = int(p_src_frame_count * src_data.src_ratio);

		src_data.end_of_input = 0;
		int error = src_simple(&src_data, interpolator_type, 1);
		if (error != 0) {
			ERR_PRINT(String(src_strerror(error)));
			return 0;
		}
		return src_data.output_frames_gen;
	} else {
		memcpy(p_dst, p_src,
				static_cast<size_t>(p_src_frame_count) * sizeof(float));
		return p_src_frame_count;
	}
}

void _vector2_array_to_float_array(const uint32_t &p_mix_frame_count,
		const Vector2 *p_process_buffer_in,
		float *p_process_buffer_out) {
	for (size_t i = 0; i < p_mix_frame_count; i++) {
		p_process_buffer_out[i] = (p_process_buffer_in[i].x + p_process_buffer_in[i].y) / 2.0;
	}
}

void _high_pass_filter(PackedFloat32Array &data, float cutoff, float sample_rate) {
	const float rc = 1.0f / (2.0f * Math_PI * cutoff);
	const float dt = 1.0f / sample_rate;
	const float alpha = dt / (rc + dt);

	float y = data[0];

	for (size_t i = 1; i < data.size(); i++) {
		y = alpha * (y + data[i] - data[i - 1]);
		data[i] = y;
	}
}

/** Check if speech is ending. */
bool _vad_simple(PackedFloat32Array &pcmf32, int sample_rate, int last_ms, float vad_thold, float freq_thold, bool verbose) {
	const int n_samples = pcmf32.size();
	const int n_samples_last = (sample_rate * last_ms) / 1000;

	if (n_samples_last >= n_samples) {
		// not enough samples - assume no speech
		return false;
	}

	if (freq_thold > 0.0f) {
		_high_pass_filter(pcmf32, freq_thold, sample_rate);
	}

	float energy_all = 0.0f;
	float energy_last = 0.0f;

	for (int i = 0; i < n_samples; i++) {
		energy_all += fabsf(pcmf32[i]);

		if (i >= n_samples - n_samples_last) {
			energy_last += fabsf(pcmf32[i]);
		}
	}

	energy_all /= n_samples;
	if (n_samples_last != 0) {
		energy_last /= n_samples_last;
	}

	if (verbose) {
		UtilityFunctions::print(rtos(energy_all), " ", rtos(energy_last), " ", rtos(vad_thold), " ", rtos(freq_thold));
	}

	if (!(energy_all < 0.0001f && energy_last < 0.0001f) || energy_last > vad_thold * energy_all) {
		return false;
	}
	return true;
}

SpeechToText::SpeechToText() {
}

void SpeechToText::set_language(int p_language) {
	language = (Language)p_language;
}

int SpeechToText::get_language() {
	return language;
}

const char *SpeechToText::_language_to_code(Language language) {
	switch (language) {
		case Auto:
			return "auto";
		case English:
			return "en";
		case Chinese:
			return "zh";
		case German:
			return "de";
		case Spanish:
			return "es";
		case Russian:
			return "ru";
		case Korean:
			return "ko";
		case French:
			return "fr";
		case Japanese:
			return "ja";
		case Portuguese:
			return "pt";
		case Turkish:
			return "tr";
		case Polish:
			return "pl";
		case Catalan:
			return "ca";
		case Dutch:
			return "nl";
		case Arabic:
			return "ar";
		case Swedish:
			return "sv";
		case Italian:
			return "it";
		case Indonesian:
			return "id";
		case Hindi:
			return "hi";
		case Finnish:
			return "fi";
		case Vietnamese:
			return "vi";
		case Hebrew:
			return "he";
		case Ukrainian:
			return "uk";
		case Greek:
			return "el";
		case Malay:
			return "ms";
		case Czech:
			return "cs";
		case Romanian:
			return "ro";
		case Danish:
			return "da";
		case Hungarian:
			return "hu";
		case Tamil:
			return "ta";
		case Norwegian:
			return "no";
		case Thai:
			return "th";
		case Urdu:
			return "ur";
		case Croatian:
			return "hr";
		case Bulgarian:
			return "bg";
		case Lithuanian:
			return "lt";
		case Latin:
			return "la";
		case Maori:
			return "mi";
		case Malayalam:
			return "ml";
		case Welsh:
			return "cy";
		case Slovak:
			return "sk";
		case Telugu:
			return "te";
		case Persian:
			return "fa";
		case Latvian:
			return "lv";
		case Bengali:
			return "bn";
		case Serbian:
			return "sr";
		case Azerbaijani:
			return "az";
		case Slovenian:
			return "sl";
		case Kannada:
			return "kn";
		case Estonian:
			return "et";
		case Macedonian:
			return "mk";
		case Breton:
			return "br";
		case Basque:
			return "eu";
		case Icelandic:
			return "is";
		case Armenian:
			return "hy";
		case Nepali:
			return "ne";
		case Mongolian:
			return "mn";
		case Bosnian:
			return "bs";
		case Kazakh:
			return "kk";
		case Albanian:
			return "sq";
		case Swahili:
			return "sw";
		case Galician:
			return "gl";
		case Marathi:
			return "mr";
		case Punjabi:
			return "pa";
		case Sinhala:
			return "si";
		case Khmer:
			return "km";
		case Shona:
			return "sn";
		case Yoruba:
			return "yo";
		case Somali:
			return "so";
		case Afrikaans:
			return "af";
		case Occitan:
			return "oc";
		case Georgian:
			return "ka";
		case Belarusian:
			return "be";
		case Tajik:
			return "tg";
		case Sindhi:
			return "sd";
		case Gujarati:
			return "gu";
		case Amharic:
			return "am";
		case Yiddish:
			return "yi";
		case Lao:
			return "lo";
		case Uzbek:
			return "uz";
		case Faroese:
			return "fo";
		case Haitian_Creole:
			return "ht";
		case Pashto:
			return "ps";
		case Turkmen:
			return "tk";
		case Nynorsk:
			return "nn";
		case Maltese:
			return "mt";
		case Sanskrit:
			return "sa";
		case Luxembourgish:
			return "lb";
		case Myanmar:
			return "my";
		case Tibetan:
			return "bo";
		case Tagalog:
			return "tl";
		case Malagasy:
			return "mg";
		case Assamese:
			return "as";
		case Tatar:
			return "tt";
		case Hawaiian:
			return "haw";
		case Lingala:
			return "ln";
		case Hausa:
			return "ha";
		case Bashkir:
			return "ba";
		case Javanese:
			return "jw";
		case Sundanese:
			return "su";
		case Cantonese:
			return "yue";
		default:
			return "en"; // Default to English if unknown language
	}
}

void SpeechToText::set_language_model(Ref<WhisperResource> p_model) {
	model = p_model;
	_load_model();
}

void SpeechToText::_load_model() {
	whisper_free(context_instance);
	context_instance = nullptr;
	if (model.is_null()) {
		ERR_PRINT("Whisper model resource is null. Set language_model before transcribing.");
		return;
	}
	const String model_path = model->get_file();
	if (model_path.is_empty()) {
		ERR_PRINT("Whisper model file path is empty. Set language_model to a .bin file.");
		return;
	}
	if (!FileAccess::file_exists(model_path)) {
		ERR_PRINT("Whisper model file not found: " + model_path);
		return;
	}
	whisper_context_params context_params = whisper_context_default_params();
	context_params.use_gpu = _is_use_gpu();
	context_params.flash_attn = flash_attn;
	UtilityFunctions::print(whisper_print_system_info());
	GodotModelLoaderContext loader_context;
	whisper_model_loader loader = {};
	if (!_open_model_loader(model_path, loader_context, loader)) {
		return;
	}
	context_instance = whisper_init_with_params(&loader, context_params);
	if (!context_instance) {
		ERR_PRINT("Failed to initialize Whisper context from model: " + model_path);
	}
}

void SpeechToText::_load_vad_model() {
	whisper_vad_free(vad_context);
	vad_context = nullptr;
	if (vad_model.is_null()) {
		return;
	}
	const String model_path = vad_model->get_file();
	if (model_path.is_empty()) {
		return;
	}
	if (!_is_valid_silero_vad_model_file(model_path)) {
		return;
	}
	whisper_vad_context_params vad_ctx_params = whisper_vad_default_context_params();
	// Silero VAD is small; current upstream VAD graph can abort on Metal during init.
	vad_ctx_params.use_gpu = false;
	GodotModelLoaderContext loader_context;
	whisper_model_loader loader = {};
	if (!_open_model_loader(model_path, loader_context, loader)) {
		return;
	}
	vad_context = whisper_vad_init_with_params(&loader, vad_ctx_params);
	if (!vad_context) {
		ERR_PRINT("Failed to load VAD model from: " + model_path);
	}
}

whisper_vad_params SpeechToText::_get_silero_vad_params() const {
	whisper_vad_params vad_params = whisper_vad_default_params();
	vad_params.threshold = vad_threshold;
	vad_params.min_speech_duration_ms = vad_min_speech_duration_ms;
	vad_params.min_silence_duration_ms = vad_min_silence_duration_ms;
	vad_params.max_speech_duration_s = vad_max_speech_duration_s > 0.0f ? vad_max_speech_duration_s : FLT_MAX;
	vad_params.speech_pad_ms = vad_speech_pad_ms;
	vad_params.samples_overlap = vad_samples_overlap;
	return vad_params;
}

PackedFloat32Array SpeechToText::_filter_speech_samples(PackedFloat32Array buffer) {
	PackedFloat32Array filtered;
	last_speech_segments.clear();
	if (!vad_context) {
		_load_vad_model();
		if (!vad_context) {
			return filtered;
		}
	}

	whisper_vad_params vad_params = _get_silero_vad_params();
	whisper_vad_segments *segments = whisper_vad_segments_from_samples(
			vad_context, vad_params, buffer.ptr(), buffer.size());
	if (!segments) {
		ERR_PRINT("Failed to detect speech segments.");
		return filtered;
	}

	const int n_segments = whisper_vad_segments_n_segments(segments);
	if (n_segments <= 0) {
		whisper_vad_free_segments(segments);
		return filtered;
	}

	const int n_samples = buffer.size();
	const int overlap_samples = vad_params.samples_overlap * WHISPER_SAMPLE_RATE;
	const int silence_samples = 0.1f * WHISPER_SAMPLE_RATE;
	std::vector<std::pair<int, int>> ranges;
	ranges.reserve(n_segments);

	for (int i = 0; i < n_segments; i++) {
		const float segment_start_cs = whisper_vad_segments_get_segment_t0(segments, i);
		const float segment_end_cs = whisper_vad_segments_get_segment_t1(segments, i);
		int segment_start = _centiseconds_to_samples(segment_start_cs);
		int segment_end = _centiseconds_to_samples(segment_end_cs);
		if (i < n_segments - 1) {
			segment_end += overlap_samples;
		}
		segment_start = std::max(0, std::min(segment_start, n_samples));
		segment_end = std::max(segment_start, std::min(segment_end, n_samples));
		if (segment_end <= segment_start) {
			continue;
		}
		Dictionary segment;
		segment["start"] = segment_start_cs;
		segment["end"] = segment_end_cs;
		last_speech_segments.push_back(segment);
		ranges.push_back({ segment_start, segment_end });
	}

	whisper_vad_free_segments(segments);
	if (ranges.empty()) {
		return filtered;
	}

	int total_samples = 0;
	for (const std::pair<int, int> &range : ranges) {
		total_samples += range.second - range.first;
	}
	total_samples += ((int)ranges.size() - 1) * silence_samples;
	filtered.resize(total_samples);
	float *dst = filtered.ptrw();
	const float *src = buffer.ptr();
	int offset = 0;
	for (int i = 0; i < (int)ranges.size(); i++) {
		const int segment_start = ranges[i].first;
		const int segment_len = ranges[i].second - ranges[i].first;
		std::memcpy(dst + offset, src + segment_start, segment_len * sizeof(float));
		offset += segment_len;
		if (i < (int)ranges.size() - 1) {
			std::memset(dst + offset, 0, silence_samples * sizeof(float));
			offset += silence_samples;
		}
	}

	return filtered;
}

void SpeechToText::set_vad_model(Ref<WhisperResource> p_model) {
	vad_model = p_model;
	last_speech_segments.clear();
	whisper_vad_free(vad_context);
	vad_context = nullptr;
}

void SpeechToText::set_vad_threshold(float p_threshold) {
	vad_threshold = std::max(0.0f, std::min(p_threshold, 1.0f));
}

void SpeechToText::set_vad_min_speech_duration_ms(int p_ms) {
	vad_min_speech_duration_ms = std::max(p_ms, 0);
}

void SpeechToText::set_vad_min_silence_duration_ms(int p_ms) {
	vad_min_silence_duration_ms = std::max(p_ms, 0);
}

void SpeechToText::set_vad_max_speech_duration_s(float p_seconds) {
	vad_max_speech_duration_s = std::max(p_seconds, 0.0f);
}

void SpeechToText::set_vad_speech_pad_ms(int p_ms) {
	vad_speech_pad_ms = std::max(p_ms, 0);
}

void SpeechToText::set_vad_samples_overlap(float p_seconds) {
	vad_samples_overlap = std::max(p_seconds, 0.0f);
}

void SpeechToText::set_token_timestamps(bool p_enable) {
	token_timestamps = p_enable;
}

void SpeechToText::set_flash_attn(bool p_enable) {
	flash_attn = p_enable;
	_load_model();
}

bool SpeechToText::get_flash_attn() {
	return flash_attn;
}

SpeechToText::~SpeechToText() {
	whisper_free(context_instance);
	context_instance = nullptr;
	whisper_vad_free(vad_context);
	vad_context = nullptr;
}

PackedFloat32Array SpeechToText::resample(PackedVector2Array buffer, SpeechToText::InterpolatorType interpolator_type) {
	int64_t buffer_len = buffer.size();
	float *buffer_float = (float *)memalloc(sizeof(float) * buffer_len);
	uint32_t expected_size = buffer_len * WHISPER_SAMPLE_RATE / AudioServer::get_singleton()->get_mix_rate();
	float *resampled_float = (float *)memalloc(sizeof(float) * expected_size);
	_vector2_array_to_float_array(buffer_len, buffer.ptr(), buffer_float);
	// Speaker frame.
	uint32_t result_size = _resample_audio_buffer(
			buffer_float, // Pointer to source buffer
			buffer_len, // Size of source buffer * sizeof(float)
			AudioServer::get_singleton()->get_mix_rate(), // Source sample rate
			WHISPER_SAMPLE_RATE, // Target sample rate
			resampled_float,
			interpolator_type);
	if (result_size != expected_size) {
		ERR_PRINT("size differ exp: " + rtos(expected_size) + " res: " + rtos(result_size));
	}
	PackedFloat32Array array;
	array.resize(result_size);
	std::memcpy(array.ptrw(), resampled_float, result_size * sizeof(float));
	memfree(buffer_float);
	memfree(resampled_float);
	return array;
}

bool SpeechToText::voice_activity_detection(PackedFloat32Array buffer) {
	/* VAD parameters */
	// The most recent 3s.
	const int vad_window_s = 3;
	const int n_samples_vad_window = WHISPER_SAMPLE_RATE * vad_window_s;
	// In VAD, compare the energy of the last 500ms to that of the total 3s.
	const int vad_last_ms = 500;
	const float vad_thold = _get_vad_thold();
	const float freq_thold = _get_freq_thold();
	/**
	 * Simple VAD from the "stream" example in whisper.cpp
	 * https://github.com/ggerganov/whisper.cpp/blob/231bebca7deaf32d268a8b207d15aa859e52dbbe/examples/stream/stream.cpp#L378
	 */
	/* Need enough accumulated audio to do VAD. */
	if ((int)buffer.size() >= n_samples_vad_window) {
		PackedFloat32Array pcmf32_window;
		pcmf32_window.resize(n_samples_vad_window);
		std::memcpy(pcmf32_window.ptrw(), buffer.ptr() + buffer.size() - n_samples_vad_window, n_samples_vad_window * sizeof(float));
		return _vad_simple(pcmf32_window, WHISPER_SAMPLE_RATE, vad_last_ms, vad_thold, freq_thold, false);
	}
	return false;
}

Array SpeechToText::detect_speech_segments(PackedFloat32Array buffer) {
	Array result;
	if (!vad_context) {
		_load_vad_model();
		if (!vad_context) {
			ERR_PRINT("VAD model not loaded. Set vad_model first.");
			return result;
		}
	}
	whisper_vad_params vad_params = _get_silero_vad_params();
	whisper_vad_segments *segments = whisper_vad_segments_from_samples(
			vad_context, vad_params, buffer.ptr(), buffer.size());
	if (!segments) {
		ERR_PRINT("Failed to detect speech segments.");
		return result;
	}
	int n = whisper_vad_segments_n_segments(segments);
	for (int i = 0; i < n; i++) {
		Dictionary seg;
		seg["start"] = whisper_vad_segments_get_segment_t0(segments, i);
		seg["end"] = whisper_vad_segments_get_segment_t1(segments, i);
		result.push_back(seg);
	}
	whisper_vad_free_segments(segments);
	return result;
}

Array SpeechToText::transcribe(PackedFloat32Array buffer, String initial_prompt, int audio_ctx) {
	Array return_value;
	last_speech_segments.clear();
	CharString initial_prompt_utf8 = initial_prompt.utf8();
	whisper_full_params whisper_params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
	whisper_params.language = _language_to_code(language);
	whisper_params.audio_ctx = audio_ctx;
	whisper_params.split_on_word = true;
	whisper_params.token_timestamps = token_timestamps;
	whisper_params.suppress_nst = true;
	whisper_params.single_segment = vad_model.is_null();
	whisper_params.max_tokens = _get_max_tokens();
	whisper_params.entropy_thold = _get_entropy_threshold();
	whisper_params.temperature_inc = 0.0f;
	whisper_params.initial_prompt = initial_prompt_utf8.get_data();

	PackedFloat32Array filtered_buffer;
	const float *samples = buffer.ptr();
	int n_samples = buffer.size();
	if (vad_model.is_valid()) {
		filtered_buffer = _filter_speech_samples(buffer);
		if (filtered_buffer.is_empty()) {
			return Array();
		}
		samples = filtered_buffer.ptr();
		n_samples = filtered_buffer.size();
	}

	if (!context_instance) {
		ERR_PRINT("Whisper context is null. Set language_model to a valid WhisperResource (.bin) before transcribing.");
		return Array();
	}
	int ret = whisper_full(context_instance, whisper_params, samples, n_samples);
	if (ret != 0) {
		ERR_PRINT("Failed to process audio, returned " + rtos(ret));
		return Array();
	}
	const int n_segments = whisper_full_n_segments(context_instance);
	String full_text;
	for (int i = 0; i < n_segments; ++i) {
		const int n_tokens = whisper_full_n_tokens(context_instance, i);
		auto segment_text = whisper_full_get_segment_text(context_instance, i);
		full_text += String::utf8(segment_text);
		for (int j = 0; j < n_tokens; j++) {
			auto token = whisper_full_get_token_data(context_instance, i, j);
			auto text = whisper_full_get_token_text(context_instance, i, j);
			Dictionary dict;
			dict["text_bytes"] = _bytes_from_c_string(text);
			dict["id"] = token.id;
			dict["p"] = token.p;
			dict["confidence"] = token.p;
			dict["plog"] = token.plog;
			dict["pt"] = token.pt;
			dict["ptsum"] = token.ptsum;
			dict["t0"] = token.t0;
			dict["t1"] = token.t1;
			dict["tid"] = token.tid;
			dict["vlen"] = token.vlen;
			return_value.push_back(dict);
		}
	}
	return_value.push_front(full_text);

	return return_value;
}

void SpeechToText::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_language"), &SpeechToText::get_language);
	ClassDB::bind_method(D_METHOD("set_language", "language"), &SpeechToText::set_language);
	ClassDB::bind_method(D_METHOD("get_language_model"), &SpeechToText::get_language_model);
	ClassDB::bind_method(D_METHOD("set_language_model", "model"), &SpeechToText::set_language_model);
	ClassDB::bind_method(D_METHOD("get_vad_model"), &SpeechToText::get_vad_model);
	ClassDB::bind_method(D_METHOD("set_vad_model", "model"), &SpeechToText::set_vad_model);
	ClassDB::bind_method(D_METHOD("get_vad_threshold"), &SpeechToText::get_vad_threshold);
	ClassDB::bind_method(D_METHOD("set_vad_threshold", "threshold"), &SpeechToText::set_vad_threshold);
	ClassDB::bind_method(D_METHOD("get_vad_min_speech_duration_ms"), &SpeechToText::get_vad_min_speech_duration_ms);
	ClassDB::bind_method(D_METHOD("set_vad_min_speech_duration_ms", "milliseconds"), &SpeechToText::set_vad_min_speech_duration_ms);
	ClassDB::bind_method(D_METHOD("get_vad_min_silence_duration_ms"), &SpeechToText::get_vad_min_silence_duration_ms);
	ClassDB::bind_method(D_METHOD("set_vad_min_silence_duration_ms", "milliseconds"), &SpeechToText::set_vad_min_silence_duration_ms);
	ClassDB::bind_method(D_METHOD("get_vad_max_speech_duration_s"), &SpeechToText::get_vad_max_speech_duration_s);
	ClassDB::bind_method(D_METHOD("set_vad_max_speech_duration_s", "seconds"), &SpeechToText::set_vad_max_speech_duration_s);
	ClassDB::bind_method(D_METHOD("get_vad_speech_pad_ms"), &SpeechToText::get_vad_speech_pad_ms);
	ClassDB::bind_method(D_METHOD("set_vad_speech_pad_ms", "milliseconds"), &SpeechToText::set_vad_speech_pad_ms);
	ClassDB::bind_method(D_METHOD("get_vad_samples_overlap"), &SpeechToText::get_vad_samples_overlap);
	ClassDB::bind_method(D_METHOD("set_vad_samples_overlap", "seconds"), &SpeechToText::set_vad_samples_overlap);
	ClassDB::bind_method(D_METHOD("get_token_timestamps"), &SpeechToText::get_token_timestamps);
	ClassDB::bind_method(D_METHOD("set_token_timestamps", "enable"), &SpeechToText::set_token_timestamps);
	ClassDB::bind_method(D_METHOD("get_flash_attn"), &SpeechToText::get_flash_attn);
	ClassDB::bind_method(D_METHOD("set_flash_attn", "enable"), &SpeechToText::set_flash_attn);
	ClassDB::bind_method(D_METHOD("transcribe", "buffer", "initial_prompt", "audio_ctx"), &SpeechToText::transcribe);
	ClassDB::bind_method(D_METHOD("voice_activity_detection", "buffer"), &SpeechToText::voice_activity_detection);
	ClassDB::bind_method(D_METHOD("detect_speech_segments", "buffer"), &SpeechToText::detect_speech_segments);
	ClassDB::bind_method(D_METHOD("get_last_speech_segments"), &SpeechToText::get_last_speech_segments);
	ClassDB::bind_method(D_METHOD("resample", "buffer"), &SpeechToText::resample);

	BIND_ENUM_CONSTANT(SRC_SINC_BEST_QUALITY);
	BIND_ENUM_CONSTANT(SRC_SINC_MEDIUM_QUALITY);
	BIND_ENUM_CONSTANT(SRC_SINC_FASTEST);
	BIND_ENUM_CONSTANT(SRC_ZERO_ORDER_HOLD);
	BIND_ENUM_CONSTANT(SRC_LINEAR);

	BIND_ENUM_CONSTANT(Auto);
	BIND_ENUM_CONSTANT(English);
	BIND_ENUM_CONSTANT(Chinese);
	BIND_ENUM_CONSTANT(German);
	BIND_ENUM_CONSTANT(Spanish);
	BIND_ENUM_CONSTANT(Russian);
	BIND_ENUM_CONSTANT(Korean);
	BIND_ENUM_CONSTANT(French);
	BIND_ENUM_CONSTANT(Japanese);
	BIND_ENUM_CONSTANT(Portuguese);
	BIND_ENUM_CONSTANT(Turkish);
	BIND_ENUM_CONSTANT(Polish);
	BIND_ENUM_CONSTANT(Catalan);
	BIND_ENUM_CONSTANT(Dutch);
	BIND_ENUM_CONSTANT(Arabic);
	BIND_ENUM_CONSTANT(Swedish);
	BIND_ENUM_CONSTANT(Italian);
	BIND_ENUM_CONSTANT(Indonesian);
	BIND_ENUM_CONSTANT(Hindi);
	BIND_ENUM_CONSTANT(Finnish);
	BIND_ENUM_CONSTANT(Vietnamese);
	BIND_ENUM_CONSTANT(Hebrew);
	BIND_ENUM_CONSTANT(Ukrainian);
	BIND_ENUM_CONSTANT(Greek);
	BIND_ENUM_CONSTANT(Malay);
	BIND_ENUM_CONSTANT(Czech);
	BIND_ENUM_CONSTANT(Romanian);
	BIND_ENUM_CONSTANT(Danish);
	BIND_ENUM_CONSTANT(Hungarian);
	BIND_ENUM_CONSTANT(Tamil);
	BIND_ENUM_CONSTANT(Norwegian);
	BIND_ENUM_CONSTANT(Thai);
	BIND_ENUM_CONSTANT(Urdu);
	BIND_ENUM_CONSTANT(Croatian);
	BIND_ENUM_CONSTANT(Bulgarian);
	BIND_ENUM_CONSTANT(Lithuanian);
	BIND_ENUM_CONSTANT(Latin);
	BIND_ENUM_CONSTANT(Maori);
	BIND_ENUM_CONSTANT(Malayalam);
	BIND_ENUM_CONSTANT(Welsh);
	BIND_ENUM_CONSTANT(Slovak);
	BIND_ENUM_CONSTANT(Telugu);
	BIND_ENUM_CONSTANT(Persian);
	BIND_ENUM_CONSTANT(Latvian);
	BIND_ENUM_CONSTANT(Bengali);
	BIND_ENUM_CONSTANT(Serbian);
	BIND_ENUM_CONSTANT(Azerbaijani);
	BIND_ENUM_CONSTANT(Slovenian);
	BIND_ENUM_CONSTANT(Kannada);
	BIND_ENUM_CONSTANT(Estonian);
	BIND_ENUM_CONSTANT(Macedonian);
	BIND_ENUM_CONSTANT(Breton);
	BIND_ENUM_CONSTANT(Basque);
	BIND_ENUM_CONSTANT(Icelandic);
	BIND_ENUM_CONSTANT(Armenian);
	BIND_ENUM_CONSTANT(Nepali);
	BIND_ENUM_CONSTANT(Mongolian);
	BIND_ENUM_CONSTANT(Bosnian);
	BIND_ENUM_CONSTANT(Kazakh);
	BIND_ENUM_CONSTANT(Albanian);
	BIND_ENUM_CONSTANT(Swahili);
	BIND_ENUM_CONSTANT(Galician);
	BIND_ENUM_CONSTANT(Marathi);
	BIND_ENUM_CONSTANT(Punjabi);
	BIND_ENUM_CONSTANT(Sinhala);
	BIND_ENUM_CONSTANT(Khmer);
	BIND_ENUM_CONSTANT(Shona);
	BIND_ENUM_CONSTANT(Yoruba);
	BIND_ENUM_CONSTANT(Somali);
	BIND_ENUM_CONSTANT(Afrikaans);
	BIND_ENUM_CONSTANT(Occitan);
	BIND_ENUM_CONSTANT(Georgian);
	BIND_ENUM_CONSTANT(Belarusian);
	BIND_ENUM_CONSTANT(Tajik);
	BIND_ENUM_CONSTANT(Sindhi);
	BIND_ENUM_CONSTANT(Gujarati);
	BIND_ENUM_CONSTANT(Amharic);
	BIND_ENUM_CONSTANT(Yiddish);
	BIND_ENUM_CONSTANT(Lao);
	BIND_ENUM_CONSTANT(Uzbek);
	BIND_ENUM_CONSTANT(Faroese);
	BIND_ENUM_CONSTANT(Haitian_Creole);
	BIND_ENUM_CONSTANT(Pashto);
	BIND_ENUM_CONSTANT(Turkmen);
	BIND_ENUM_CONSTANT(Nynorsk);
	BIND_ENUM_CONSTANT(Maltese);
	BIND_ENUM_CONSTANT(Sanskrit);
	BIND_ENUM_CONSTANT(Luxembourgish);
	BIND_ENUM_CONSTANT(Myanmar);
	BIND_ENUM_CONSTANT(Tibetan);
	BIND_ENUM_CONSTANT(Tagalog);
	BIND_ENUM_CONSTANT(Malagasy);
	BIND_ENUM_CONSTANT(Assamese);
	BIND_ENUM_CONSTANT(Tatar);
	BIND_ENUM_CONSTANT(Hawaiian);
	BIND_ENUM_CONSTANT(Lingala);
	BIND_ENUM_CONSTANT(Hausa);
	BIND_ENUM_CONSTANT(Bashkir);
	BIND_ENUM_CONSTANT(Javanese);
	BIND_ENUM_CONSTANT(Sundanese);
	BIND_ENUM_CONSTANT(Cantonese);

	BIND_ENUM_CONSTANT(SPEECH_SETTING_SAMPLE_RATE);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "language", PROPERTY_HINT_ENUM, "Auto,English,Chinese,German,Spanish,Russian,Korean,French,Japanese,Portuguese,Turkish,Polish,Catalan,Dutch,Arabic,Swedish,Italian,Indonesian,Hindi,Finnish,Vietnamese,Hebrew,Ukrainian,Greek,Malay,Czech,Romanian,Danish,Hungarian,Tamil,Norwegian,Thai,Urdu,Croatian,Bulgarian,Lithuanian,Latin,Maori,Malayalam,Welsh,Slovak,Telugu,Persian,Latvian,Bengali,Serbian,Azerbaijani,Slovenian,Kannada,Estonian,Macedonian,Breton,Basque,Icelandic,Armenian,Nepali,Mongolian,Bosnian,Kazakh,Albanian,Swahili,Galician,Marathi,Punjabi,Sinhala,Khmer,Shona,Yoruba,Somali,Afrikaans,Occitan,Georgian,Belarusian,Tajik,Sindhi,Gujarati,Amharic,Yiddish,Lao,Uzbek,Faroese,Haitian_Creole,Pashto,Turkmen,Nynorsk,Maltese,Sanskrit,Luxembourgish,Myanmar,Tibetan,Tagalog,Malagasy,Assamese,Tatar,Hawaiian,Lingala,Hausa,Bashkir,Javanese,Sundanese,Cantonese"), "set_language", "get_language");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "language_model", PROPERTY_HINT_RESOURCE_TYPE, "WhisperResource"), "set_language_model", "get_language_model");
	ADD_GROUP("Voice Activity Detection", "vad_");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "vad_model", PROPERTY_HINT_RESOURCE_TYPE, "WhisperResource"), "set_vad_model", "get_vad_model");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "vad_threshold", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_vad_threshold", "get_vad_threshold");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "vad_min_speech_duration_ms", PROPERTY_HINT_RANGE, "0,60000,1,or_greater"), "set_vad_min_speech_duration_ms", "get_vad_min_speech_duration_ms");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "vad_min_silence_duration_ms", PROPERTY_HINT_RANGE, "0,60000,1,or_greater"), "set_vad_min_silence_duration_ms", "get_vad_min_silence_duration_ms");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "vad_max_speech_duration_s", PROPERTY_HINT_RANGE, "0,3600,0.1,or_greater"), "set_vad_max_speech_duration_s", "get_vad_max_speech_duration_s");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "vad_speech_pad_ms", PROPERTY_HINT_RANGE, "0,5000,1,or_greater"), "set_vad_speech_pad_ms", "get_vad_speech_pad_ms");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "vad_samples_overlap", PROPERTY_HINT_RANGE, "0,10,0.01,or_greater"), "set_vad_samples_overlap", "get_vad_samples_overlap");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "token_timestamps"), "set_token_timestamps", "get_token_timestamps");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "flash_attn"), "set_flash_attn", "get_flash_attn");
}
