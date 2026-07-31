const express = require("express");
const cors = require("cors");
const multer = require("multer");
const fs = require("fs");
const path = require("path");
const { exec } = require("child_process");

const app = express();
const PORT = process.env.PORT || 5001;

app.use(cors());
app.use(express.json({ limit: "20mb" }));
app.use(express.urlencoded({ extended: true, limit: "20mb" }));

const uploadDir = path.join(__dirname, "uploads");
if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir, { recursive: true });

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, uploadDir),
  filename: (req, file, cb) => cb(null, `upload_${Date.now()}.png`)
});
const upload = multer({ storage });

const PROJECT_ROOT = path.resolve(__dirname, "../../");

function runPython(cmd) {
  return new Promise((resolve, reject) => {
    exec(cmd, {
      cwd: PROJECT_ROOT,
      env: { ...process.env, PYTHONIOENCODING: "utf-8" }
    }, (error, stdout, stderr) => {
      if (error) return reject(new Error(stderr || error.message));
      try {
        const lines = stdout.trim().split("\n");
        const jsonLine = lines.find(l => l.startsWith("{")) || lines[lines.length - 1];
        resolve(JSON.parse(jsonLine));
      } catch (e) {
        reject(new Error("Failed to parse model output"));
      }
    });
  });
}

// Health check
app.get("/api/health", (req, res) => {
  const weightsExists = fs.existsSync(path.join(PROJECT_ROOT, "checkpoints", "best_cnn_model_weights.pth"))
    || fs.existsSync(path.join(PROJECT_ROOT, "best_cnn_model_weights.pth"));
  res.json({
    status: "online",
    modelAvailable: weightsExists,
    activeCheckpoint: weightsExists ? "best_cnn_model_weights.pth" : "not trained yet",
    timestamp: new Date().toISOString()
  });
});

// Predict image (canvas base64 or file upload)
app.post("/api/predict/image", upload.single("image"), async (req, res) => {
  let tempFile = null;
  try {
    const mode = req.body.mode || "char";

    if (req.file) {
      tempFile = req.file.path;
    } else if (req.body.imageBase64) {
      const b64 = req.body.imageBase64.replace(/^data:image\/\w+;base64,/, "");
      tempFile = path.join(uploadDir, `canvas_${Date.now()}.png`);
      fs.writeFileSync(tempFile, Buffer.from(b64, "base64"));
    } else {
      return res.status(400).json({ error: "No image provided" });
    }

    const script = path.join(PROJECT_ROOT, "predict.py");
    const cmd = `python3 "${script}" --image "${tempFile}" --mode ${mode} --json`;
    const prediction = await runPython(cmd);
    res.json({ success: true, prediction });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  } finally {
    if (tempFile && fs.existsSync(tempFile)) {
      try { fs.unlinkSync(tempFile); } catch (e) {}
    }
  }
});

// Predict custom synthetic text using already trained model
app.post("/api/predict/synthetic_text", async (req, res) => {
  let tempFiles = [];
  try {
    const { text } = req.body;
    if (!text || !text.trim()) {
      return res.status(400).json({ error: "No text provided for synthetic prediction" });
    }

    const script = path.join(PROJECT_ROOT, "predict.py");
    const pyCode = `
import json, sys, os, cv2
from synthetic_word_generator import SyntheticWordGenerator, extract_words
from predict import BanglaPredictor

txt = ${JSON.stringify(text)}
words = extract_words(txt)[:12]
gen = SyntheticWordGenerator()
predictor = BanglaPredictor()

results = []
for i, w in enumerate(words):
    img, gt = gen.generate_word_image(w)
    tmp_p = f"temp_synth_{i}_{Date.now() if 'Date' in locals() else 0}.png"
    cv2.imwrite(tmp_p, img)
    try:
        pred_res = predictor.predict_file(tmp_p)
        results.append({
            "gt": gt,
            "prediction": pred_res.get("prediction", ""),
            "confidence": pred_res.get("confidence", 0.0),
            "match": gt == pred_res.get("prediction", ""),
            "img_b64": pred_res.get("img_b64", "")
        })
    finally:
        if os.path.exists(tmp_p):
            try: os.remove(tmp_p)
            except: pass

print(json.dumps({"success": True, "results": results}, ensure_ascii=False))
`;

    const pyFile = path.join(PROJECT_ROOT, `temp_eval_${Date.now()}.py`);
    fs.writeFileSync(pyFile, pyCode, "utf-8");

    exec(`python3 "${pyFile}"`, { cwd: PROJECT_ROOT, env: { ...process.env, PYTHONIOENCODING: "utf-8" } }, (error, stdout, stderr) => {
      if (fs.existsSync(pyFile)) {
        try { fs.unlinkSync(pyFile); } catch (e) {}
      }
      if (error) {
        return res.status(500).json({ success: false, error: stderr || error.message });
      }
      try {
        const jsonOutput = JSON.parse(stdout.trim());
        res.json(jsonOutput);
      } catch (e) {
        res.status(500).json({ success: false, error: "Failed to parse evaluation output" });
      }
    });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

// Train on custom synthetic Bangla text input
app.post("/api/train/synthetic", async (req, res) => {
  try {
    const { text, epochs = 3 } = req.body;
    if (!text || !text.trim()) {
      return res.status(400).json({ error: "No text provided for synthetic training" });
    }

    const textPath = path.join(PROJECT_ROOT, "temp_custom_text.txt");
    fs.writeFileSync(textPath, text.trim(), "utf-8");

    const script = path.join(PROJECT_ROOT, "train_rl.py");
    const cmd = `PYTORCH_ENABLE_MPS_FALLBACK=1 python3 "${script}" --text-file "${textPath}" --epochs ${epochs}`;
    
    exec(cmd, { cwd: PROJECT_ROOT, env: { ...process.env, PYTHONIOENCODING: "utf-8" } }, (error, stdout, stderr) => {
      if (textPath && fs.existsSync(textPath)) {
        try { fs.unlinkSync(textPath); } catch (e) {}
      }
      if (error) {
        return res.status(500).json({ success: false, error: stderr || error.message });
      }
      res.json({ success: true, message: `Successfully trained synthetic model for ${epochs} epoch(s)!`, logs: stdout });
    });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

app.listen(PORT, () => {
  console.log(`Bangla HTR API running on http://localhost:${PORT}`);
});
