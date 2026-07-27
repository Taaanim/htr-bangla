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

app.listen(PORT, () => {
  console.log(`Bangla HTR API running on http://localhost:${PORT}`);
});
