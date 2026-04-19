#ifndef WHISPER_RESOURCE_H
#define WHISPER_RESOURCE_H

#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/core/class_db.hpp>

using namespace godot;

class WhisperResource : public Resource {
	GDCLASS(WhisperResource, Resource);

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_file", "path"), &WhisperResource::set_file);
		ClassDB::bind_method(D_METHOD("get_file"), &WhisperResource::get_file);
		ADD_PROPERTY(PropertyInfo(Variant::STRING, "file", PROPERTY_HINT_FILE, "*.bin"), "set_file", "get_file");

		ClassDB::bind_method(D_METHOD("get_content"), &WhisperResource::get_content);
	}

	String file;

public:
	void set_file(const String &p_file) {
		file = p_file;
		emit_changed();
	}

	String get_file() const {
		return file;
	}

	PackedByteArray get_content();
	WhisperResource() {}
	~WhisperResource() {}
};
#endif // WHISPER_RESOURCE_H
