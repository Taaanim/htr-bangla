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

// Upload directory setup
const uploadDir = path.join(__dirname, "uploads");
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, uploadDir),
  filename: (req, file, cb) => cb(null, `upload_${Date.now()}_${file.originalname || "image.png"}`)
});

const upload = multer({ storage });

// Project Root Directory (one level up from web_app/server)
const PROJECT_ROOT = path.resolve(__dirname, "../../");

/**
 * Executes python predict.py command and returns JSON prediction
 */
function runPythonPrediction(imagePath) {
  return new Promise((resolve, reject) => {
    const scriptPath = path.join(PROJECT_ROOT, "predict.py");
    const cmd = `python3 "${scriptPath}" --image "${imagePath}" --json`;

    exec(cmd, { cwd: PROJECT_ROOT }, (error, stdout, stderr) => {
      if (error) {
        console.error("Python Exec Error:", stderr || error.message);
        return reject(new Error(stderr || error.message));
      }
      try {
        // Extract JSON string from stdout
        const lines = stdout.trim().split("\n");
        const jsonLine = lines.find(l => l.startsWith("{") && l.endsWith("}")) || lines[lines.length - 1];
        const parsed = JSON.parse(jsonLine);
        resolve(parsed);
      } catch (parseErr) {
        console.error("JSON Parse Error:", stdout);
        reject(new Error("Failed to parse prediction output from model."));
      }
    });
  });
}

// 1. Healthcheck Route
app.get("/api/health", (req, res) => {
  const checkpointsDir = path.join(PROJECT_ROOT, "checkpoints");
  const finalModel = path.join(checkpointsDir, "final_htr_model.pth");
  const finetuneModel = path.join(checkpointsDir, "finetune_model.pth");

  const modelAvailable = fs.existsSync(finalModel) || fs.existsSync(finetuneModel);

  res.json({
    status: "online",
    modelAvailable,
    activeCheckpoint: fs.existsSync(finalModel) ? "final_htr_model.pth" : (fs.existsSync(finetuneModel) ? "finetune_model.pth" : "initializing"),
    timestamp: new Date().toISOString()
  });
});

// 2. Predict Image File Upload Route
app.post("/api/predict/image", upload.single("image"), async (req, res) => {
  let tempFilePath = null;
  try {
    if (req.file) {
      tempFilePath = req.file.path;
    } else if (req.body.imageBase64) {
      // Base64 canvas data
      const base64Data = req.body.imageBase64.replace(/^data:image\/\w+;base64,/, "");
      tempFilePath = path.join(uploadDir, `canvas_${Date.now()}.png`);
      fs.writeFileSync(tempFilePath, Buffer.from(base64Data, "base64"));
    } else {
      return res.status(400).json({ error: "No image file or base64 data provided." });
    }

    const prediction = await runPythonPrediction(tempFilePath);

    res.json({
      success: true,
      prediction
    });
  } catch (err) {
    console.error("Prediction Route Error:", err);
    res.status(500).json({ success: false, error: err.message });
  } finally {
    if (tempFilePath && fs.existsSync(tempFilePath)) {
      try { fs.unlinkSync(tempFilePath); } catch (e) {}
    }
  }
});

// 3. Predict Synthetic Text Route
app.post("/api/predict/text", async (req, res) => {
  const { text } = req.body;
  if (!text || typeof text !== "string") {
    return res.status(400).json({ error: "Valid Bangla text string is required." });
  }

  // Create temporary synthetic image via python script snippet
  const tempImgPath = path.join(uploadDir, `synth_${Date.now()}.png`);
  const pythonScript = `
from dataset import BanglaSyntheticTextGenerator
gen = BanglaSyntheticTextGenerator()
img = gen.render_text("${text.replace(/"/g, '\\"')}")
img.save("${tempImgPath}")
`;

  const scriptFile = path.join(uploadDir, `render_${Date.now()}.py`);
  fs.writeFileSync(scriptFile, pythonScript);

  exec(`python3 "${scriptFile}"`, { cwd: PROJECT_ROOT }, async (err) => {
    try { fs.unlinkSync(scriptFile); } catch (e) {}

    if (err || !fs.existsSync(tempImgPath)) {
      return res.status(500).json({ error: "Failed to render synthetic text image." });
    }

    try {
      const prediction = await runPythonPrediction(tempImgPath);
      res.json({ success: true, textInput: text, prediction });
    } catch (predErr) {
      res.status(500).json({ error: predErr.message });
    } finally {
      if (fs.existsSync(tempImgPath)) {
        try { fs.unlinkSync(tempImgPath); } catch (e) {}
      }
    }
  });
});

app.listen(PORT, () => {
  console.log(`Bangla HTR API Express Server running on http://localhost:${PORT}`);
});
