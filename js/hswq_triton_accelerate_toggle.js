import { app } from "../../scripts/app.js";

/** Same pattern as sdxl_lora_dynamic_v3.js: BOOLEAN backend + widget.type = "toggle". */
const NODE_IDS = new Set([
    "HSWQFP8E4M3UNetLoader",
    "NunchakuUssoewwinCheckpointLoaderSDXL",
]);
const WIDGET_NAME = "triton_accelerate";
const WIDGET_LABEL = "Triton accelerate";

function forceTritonToggle(node) {
    if (!node || !NODE_IDS.has(node.comfyClass)) return;
    const widgets = node.widgets || [];
    const w = widgets.find((x) => x.name === WIDGET_NAME);
    if (!w) return;
    w.type = "toggle";
    w.label = WIDGET_LABEL;
    if (w.computeSize) delete w.computeSize;
}

app.registerExtension({
    name: "nunchaku_ussoewwin.hswq_triton_accelerate_toggle",

    nodeCreated(node) {
        forceTritonToggle(node);
    },

    loadedGraphNode(node) {
        forceTritonToggle(node);
    },
});
