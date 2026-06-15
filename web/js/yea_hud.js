// ComfyUI-YEAHUD — OPTIONAL frontend UX.
// Rendering works fully without this file. Everything is wrapped so a failure
// here can never break the backend node or the graph.
import { app } from "../../scripts/app.js";

const PALETTE = ["#FF2A1A", "#FFB0DF", "#225CFF", "#8DFF28"];

app.registerExtension({
  name: "YEA.HUD",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    try {
      if (!nodeData || nodeData.category !== "YEA/HUD") return;

      // Brand tint for YEA/HUD nodes.
      const onCreate = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const r = onCreate ? onCreate.apply(this, arguments) : undefined;
        try {
          this.color = "#1f1530";
          this.bgcolor = "#140d20";
        } catch (e) { /* non-fatal */ }
        return r;
      };

      // Renderer: a tiny palette swatch + "Test first frame" hint in the context menu.
      if (nodeData.name === "YEAHUDRenderer") {
        const getMenu = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (_, options) {
          try {
            options.push({
              content: "YEA HUD — default palette",
              disabled: true,
            });
            options.push({
              content: PALETTE.join("  "),
              disabled: true,
            });
          } catch (e) { /* non-fatal */ }
          return getMenu ? getMenu.apply(this, arguments) : undefined;
        };
      }
    } catch (e) {
      console.warn("[YEA.HUD] frontend extension skipped:", e);
    }
  },
});
