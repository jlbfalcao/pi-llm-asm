// model.s — embeds the weights (PIML format) inside kernel8.img
.section .rodata.model, "a"
.balign 64
.global model_blob, model_blob_end
model_blob:
    .incbin "model.piml"
model_blob_end:
