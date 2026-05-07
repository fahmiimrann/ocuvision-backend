const express = require('express');
const cors = require('cors');
const multer = require('multer');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;
const upload = multer({ storage: multer.memoryStorage() });

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
        message: "Welcome to the Ophthalmic Intelligence Backend" 
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
        fileName: req.file.originalname
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
