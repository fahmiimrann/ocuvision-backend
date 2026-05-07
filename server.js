const express = require('express');
const cors = require('cors');
const multer = require('multer');
const fs = require('fs');
const path = require('path');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;
const upload = multer({ storage: multer.memoryStorage() });
const MODEL_DIR = path.join(__dirname, 'uploaded-models');
const MODEL_REGISTRY_FILE = path.join(MODEL_DIR, 'models.json');
const DEFAULT_MODEL = {
    id: 'demo-retina-core',
    name: 'Demo Retina Core',
    originalName: 'Built-in demo backend logic',
    type: 'demo',
    size: 0,
    uploadedAt: '2026-05-07T00:00:00.000Z',
    active: true
};

function ensureModelRegistry() {
    if (!fs.existsSync(MODEL_DIR)) {
        fs.mkdirSync(MODEL_DIR, { recursive: true });
    }

    if (!fs.existsSync(MODEL_REGISTRY_FILE)) {
        fs.writeFileSync(MODEL_REGISTRY_FILE, JSON.stringify([DEFAULT_MODEL], null, 2));
    }
}

function readModels() {
    ensureModelRegistry();
    try {
        const models = JSON.parse(fs.readFileSync(MODEL_REGISTRY_FILE, 'utf8'));
        return Array.isArray(models) && models.length ? models : [DEFAULT_MODEL];
    } catch (err) {
        return [DEFAULT_MODEL];
    }
}

function writeModels(models) {
    ensureModelRegistry();
    fs.writeFileSync(MODEL_REGISTRY_FILE, JSON.stringify(models, null, 2));
}

function getActiveModel() {
    return readModels().find(model => model.active) || DEFAULT_MODEL;
}

const modelUpload = multer({
    storage: multer.diskStorage({
        destination: function (_req, _file, cb) {
            ensureModelRegistry();
            cb(null, MODEL_DIR);
        },
        filename: function (_req, file, cb) {
            const safeBase = path.basename(file.originalname).replace(/[^a-z0-9._-]/gi, '_');
            cb(null, `${Date.now()}-${safeBase}`);
        }
    }),
    limits: { fileSize: 100 * 1024 * 1024 }
});

/**
 * CORS Configuration
 * Allows your GitHub Pages frontend to communicate with this Render backend.
 */
const allowedOrigins = [
    'http://localhost:5500', // For local development testing
    'http://127.0.0.1:5500',
    'http://localhost:3000',
    'https://calisto.github.io',
    'https://fahmiimrann.github.io',
    'https://fahmiimrann.github.io/calisto.github.io',
    'https://fahmimrann.github.io',
    'https://fahmimrann.github.io/calisto.github.io' // Your live frontend
];

app.use(cors({
    origin: function (origin, callback) {
        // Allow requests with no origin (like mobile apps or curl)
        if (!origin) return callback(null, true);
        if (allowedOrigins.indexOf(origin) === -1) {
            const msg = 'The CORS policy for this site does not allow access from the specified Origin.';
            return callback(new Error(msg), false);
        }
        return callback(null, true);
    }
}));

app.use(express.json());

// Base route to verify the API is running
app.get('/', (req, res) => {
    res.json({ 
        status: "Online", 
        project: "OcuVision AI API",
        message: "Welcome to the Ophthalmic Intelligence Backend",
        activeModel: getActiveModel().name
    });
});

// Demo login endpoint used by index.html.
// Replace this with database-backed authentication before real clinical use.
app.post('/api/login', (req, res) => {
    const { username, password } = req.body;

    const demoUsers = [
        { username: '1', password: '1', type: 'intern', name: 'Fahmi' },
        { username: 'doctor', password: 'ocularxr', type: 'doctor', name: 'Dr. Julian Voss' },
        { username: 'nurse', password: 'nurs3', type: 'nurse', name: 'Nurse Meera Syed' }
    ];

    const user = demoUsers.find(item => item.username === username && item.password === password);

    if (!user) {
        return res.status(401).json({ error: 'Invalid username or password.' });
    }

    const { password: _password, ...safeUser } = user;
    res.json({ user: safeUser });
});

// Demo register endpoint. This returns the user to the frontend, but does not persist yet.
app.post('/api/register', (req, res) => {
    const { username, name, type } = req.body;

    if (!username || !name || !type) {
        return res.status(400).json({ error: 'Name, username, and type are required.' });
    }

    res.status(201).json({
        user: {
            username,
            name,
            type
        }
    });
});

// Demo AI analysis endpoint used by the diagnostics image upload flow.
app.post('/api/analyze', upload.single('image'), (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: 'No image uploaded.' });
    }

    const activeModel = getActiveModel();
    const demoResults = [
        {
            status: 'Alert',
            score: '94.8',
            result: 'Diabetic Retinopathy',
            severity: 'Critical',
            notes: 'Multiple microaneurysms and intraretinal hemorrhages detected. Expert validation recommended.'
        },
        {
            status: 'Warning',
            score: '89.3',
            result: 'Early Glaucoma indicators detected',
            severity: 'Moderate',
            notes: 'Cup-to-disc ratio and vessel morphology should be reviewed by an ophthalmologist.'
        },
        {
            status: 'Optimal',
            score: '98.7',
            result: 'No urgent retinal abnormality detected',
            severity: 'Healthy',
            notes: 'Fundus pattern is within baseline range for this demo backend screening.'
        }
    ];

    const result = demoResults[Math.floor(Math.random() * demoResults.length)];

    res.json({
        ...result,
        confidence: `${result.score}%`,
        fileName: req.file.originalname,
        model: {
            id: activeModel.id,
            name: activeModel.name
        }
    });
});

// Model registry endpoints for Preferences > AI Core.
app.get('/api/models', (_req, res) => {
    res.json({
        models: readModels(),
        activeModel: getActiveModel()
    });
});

app.post('/api/models', modelUpload.single('model'), (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: 'No model file uploaded.' });
    }

    const models = readModels().map(model => ({ ...model, active: false }));
    const model = {
        id: `model-${Date.now()}`,
        name: req.body.name?.trim() || path.parse(req.file.originalname).name,
        originalName: req.file.originalname,
        storedFile: req.file.filename,
        type: path.extname(req.file.originalname).replace('.', '').toLowerCase() || 'model',
        size: req.file.size,
        uploadedAt: new Date().toISOString(),
        active: true
    };

    models.push(model);
    writeModels(models);

    res.status(201).json({
        model,
        models,
        message: `${model.name} uploaded and activated.`
    });
});

app.post('/api/models/select', (req, res) => {
    const { id } = req.body;
    const models = readModels();

    if (!models.some(model => model.id === id)) {
        return res.status(404).json({ error: 'Model not found.' });
    }

    const updatedModels = models.map(model => ({
        ...model,
        active: model.id === id
    }));

    writeModels(updatedModels);

    res.json({
        activeModel: updatedModels.find(model => model.id === id),
        models: updatedModels
    });
});

// Example API endpoint for patient records
app.get('/api/records', (req, res) => {
    const mockRecords = [
        { id: 'REC001', patient: 'John Doe', result: 'Healthy', date: '2024-05-20' },
        { id: 'REC002', patient: 'Jane Smith', result: 'Glaucoma Risk', date: '2024-05-21' },
        { id: 'REC003', patient: 'Robert Chen', result: 'Macular Degeneration', date: '2024-05-22' }
    ];
    res.json(mockRecords);
});

app.listen(PORT, () => {
    console.log(`OcuVision Server is running on port ${PORT}`);
});
