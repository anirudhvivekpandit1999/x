const express = require("express");
const crypto = require("crypto");
const multer = require("multer");
const path = require("path");
const cors = require("cors");
const { spawn } = require("child_process");
const { default: axios } = require("axios");
const FormData = require("form-data");
const fs = require("fs");
const sequelize = require("./database");
const { secretKey } = require("./security");

const app = express();
const port = 3000;
let pythonProcess = null;

// **Start Python Server**
async function startPythonServer(callback) {
    if (pythonProcess) {
        console.log("Python server is already running.");
        return callback && callback();
    }

    const running = await isPythonServerRunning();
    if (running) {
        console.log("Python AI server is already running.");
        return callback && callback();
    }

    console.log("Starting Python AI server...");
    pythonProcess = spawn("python", ["app.py"], { stdio: "inherit" });

    pythonProcess.on("exit", (code) => {
        console.log(`Python server exited with code ${code}`);
        pythonProcess = null;
    });

    setTimeout(() => {
        if (callback) callback();
    }, 3000);
}

async function isPythonServerRunning() {
    try {
        await axios.get("http://127.0.0.1:5001/health");
        return true;
    } catch {
        return false;
    }
}

// **Send Data to Python AI**
async function sendData(data) {
    try {
        const response = await axios.post("http://127.0.0.1:5001/ai", data, {
            headers: { "Content-Type": "application/json" },
        });
        console.log("Response from AI:", response.data);
        return response.data.response;
    } catch (error) {
        if (error.code === "ECONNREFUSED") {
            console.log("Python AI server not running. Checking now...");
            startPythonServer(() => sendData(data));
        } else {
            console.error("Error sending request:", error.message);
        }
    }
}

// **Encrypt & Decrypt Functions**
const iv = Buffer.alloc(16, 0);

function encryptData(data) {
    const cipher = crypto.createCipheriv("aes-256-cbc", secretKey, iv);
    let encrypted = cipher.update(JSON.stringify(data), "utf-8", "hex");
    encrypted += cipher.final("hex");
    return encrypted;
}

function decryptData(encryptedData) {
    const decipher = crypto.createDecipheriv("aes-256-cbc", secretKey, iv);
    let decrypted = decipher.update(encryptedData, "hex", "utf-8");
    decrypted += decipher.final("utf-8");
    return JSON.parse(decrypted);
}

// **Execute MySQL Stored Procedure**
async function callStoredProcedure(procedureName, req) {
    try {
        const decryptedBody = decryptData(req.body.encryptedData);
        const replacements = Object.values(decryptedBody);

        const result = await sequelize.query(
            `CALL ${procedureName}(${replacements.map(() => "?").join(", ")})`,
            { replacements, type: sequelize.QueryTypes.RAW }
        );

        return result[0] ? encryptData(result[0]) : { message: "No data found" };
    } catch (error) {
        console.error(`Error executing ${procedureName}:`, error);
        throw new Error("Database error");
    }
}

// **Multer File Upload Setup**
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, "/path/to/upload/directory"); // Change this path
    },
    filename: (req, file, cb) => {
        cb(null, Date.now() + path.extname(file.originalname));
    },
});
const upload = multer({ storage });

// **Middlewares**
app.use(express.json());
app.use(cors());

// **Routes**
app.post("/api/getCoalProperties", async (req, res) => {
    try {
        const coalProperties = await callStoredProcedure("SP_GetCoalProperties", req);
        res.status(200).json({ coalProperties });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.post("/api/updateCoalProperties", async (req, res) => {
    try {
        const coalProperties = await callStoredProcedure("SP_UpdateCoalProperties", req);
        res.status(200).json({ coalProperties });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.post("/api/insertCoalProperties", async (req, res) => {
    try {
        const coalProperties = await callStoredProcedure("SP_InsertCoalProperties", req);
        res.status(200).json({ coalProperties });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.post("/api/deleteCoalProperties", async (req, res) => {
    try {
        const coalProperties = await callStoredProcedure("SP_DeleteCoalProperties", req);
        res.status(200).json({ encryptedData: coalProperties ?? { message: "No coal properties found" } });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.post("/api/signup", async (req, res) => {
    try {
        const createdAt = new Date().toISOString().slice(0, 19).replace("T", " ");
        const data = await callStoredProcedure("SP_SignUp", { ...req, createdAt });
        res.status(200).json({ encryptedData: data ?? { message: "Signup failed" } });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.post("/api/login", async (req, res) => {
    try {
        const user = await callStoredProcedure("SP_Login", req);
        res.status(200).json({
            encryptedData: user?.length ? user[0] : { error: "Invalid email or password" },
        });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.post("/api/costAi", async (req, res) => {
    try {
        const response = await sendData(req.body);
        res.status(200).json({ response });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.post("/api/uploadExcelTraining", upload.single("file"), async (req, res) => {
    try {
        if (!req.file) return res.status(400).send("No file uploaded.");
        const formData = new FormData();
        formData.append("file", fs.createReadStream(req.file.path), req.file.originalname);
        const result = await callStoredProcedure("sp_insertfile ", req);

        const response = await axios.post("http://127.0.0.1:5001/upload-excel", formData, {
            headers: { ...formData.getHeaders() },
        });
        fs.unlinkSync(req.file.path);
        res.send(`File uploaded successfully! Response: ${response.data}`);
    } catch (error) {
        console.error(error);
        res.status(500).send("Error uploading file.");
    }
});

app.post("/api/getblendedcoalproperties", async (req, res) => {
    try {
        const result = await callStoredProcedure("sp_getblendedcoalproperties", req);
        res.status(200).json({
            encryptedData: result,
        });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.post("/api/insertblendedcoalproperties", async (req, res) => {
    try {
        const result = await callStoredProcedure("sp_insertblendedcoalproperties ", req);
        res.status(200).json({
            encryptedData: result,
        });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.post("/api/getcokeproperties", async (req, res) => {
    try {
        const result = await callStoredProcedure("sp_getcokeproperties  ", req);
        res.status(200).json({
            encryptedData: result,
        });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.post("/api/insertcokeproperties", async (req, res) => {
    try {
        const result = await callStoredProcedure("ps_insertcokeproperties", req);
        res.status(200).json({
            encryptedData: result,
        });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});


app.post("/api/deleteexcelfile", async (req, res) => {
    try {
        const result = await callStoredProcedure("sp_deletefile", req);

        const filePath = path.join(__dirname, req.body.filePath);

        if (fs.existsSync(filePath)) {
            fs.unlinkSync(filePath); 
            console.log(`File deleted: ${filePath}`);
        } else {
            console.log(`File not found: ${filePath}`);
        }

        res.status(200).json({
            message: "File deleted successfully",
            encryptedData: result,
        });

    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.post("/api/deleteonfileid", async (req, res) => {
    try {
        const result = await callStoredProcedure("sp_deleteonfileid ", req);
        res.status(200).json({
            encryptedData: result,
        });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.post("/api/inserttrainingdata", async (req, res) => {
    try {
        const result = await callStoredProcedure("sp_insert_training_data ", req);
        res.status(200).json({
            encryptedData: result,
        });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});


// **Start Server**
app.listen(port, () => {
    console.log(`API running at http://localhost:${port}`);
    startPythonServer();
});
