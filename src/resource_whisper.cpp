#include "resource_whisper.h"

#include <godot_cpp/classes/file_access.hpp>
#include <godot_cpp/core/error_macros.hpp>

PackedByteArray WhisperResource::get_content() {
	PackedByteArray content;
	String p_path = get_file();
	if (p_path.is_empty()) {
		ERR_PRINT("Whisper model file path is empty.");
		return content;
	}
	if (!FileAccess::file_exists(p_path)) {
		ERR_PRINT("Whisper model file not found: " + p_path);
		return content;
	}
	content = FileAccess::get_file_as_bytes(p_path);
	if (content.is_empty()) {
		ERR_PRINT("Whisper model file is empty or could not be read: " + p_path);
	}
	return content;
}
