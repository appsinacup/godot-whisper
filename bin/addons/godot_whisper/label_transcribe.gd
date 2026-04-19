extends RichTextLabel

## Completed text from transcription
var completed_text := ""

## Partial text from transcription
var partial_text := ""


## Ready function to initialize the label
func _ready() -> void:
	custom_minimum_size.x = 400
	bbcode_enabled = true
	fit_content = true


## Update the text displayed on the label
func update_text() -> void:
	text = completed_text + "[color=green]" + partial_text + "[/color]"


## Process function to update text every frame
func _process(_delta: float) -> void:
	update_text()


## Handle the speech-to-text transcribed message.
## is_complete: true when the sentence is finalized, false when still in progress.
func _on_speech_to_text_transcribed_msg(is_complete: bool, new_text: String) -> void:
	if is_complete:
		completed_text += new_text
		partial_text = ""
	else:
		if new_text != "":
			partial_text = new_text
