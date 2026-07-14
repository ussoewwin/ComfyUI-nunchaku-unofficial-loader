import { app } from "../../scripts/app.js";

const NODE_ID = "NunchakuUssoewwinCheckpointLoaderSDXL";
const DISPLAY_TITLE = "HSWQ Checkpoint Loader (SDXL)";

function applyTitle(node) {
    if (!node || node.comfyClass !== NODE_ID) return;
    node.title = DISPLAY_TITLE;
    if (node.constructor) {
        node.constructor.title = DISPLAY_TITLE;
    }
}

app.registerExtension({
    name: "nunchaku_ussoewwin.hswq_checkpoint_loader_sdxl_title",

    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (nodeData?.name !== NODE_ID) return;
        nodeData.display_name = DISPLAY_TITLE;
        if (nodeType) {
            nodeType.title = DISPLAY_TITLE;
        }
    },

    nodeCreated(node) {
        applyTitle(node);
    },

    loadedGraphNode(node) {
        applyTitle(node);
    },
});
