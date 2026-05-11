// Storage abstraction.
// Uses Supabase when SUPABASE_URL + SUPABASE_SERVICE_KEY are set,
// otherwise falls back to local JSON files in `./data` so dev still works.

const fs = require('fs');
const path = require('path');
const { createClient } = require('@supabase/supabase-js');

const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_KEY = process.env.SUPABASE_SERVICE_KEY || process.env.SUPABASE_SERVICE_ROLE_KEY;
const useSupabase = !!(SUPABASE_URL && SUPABASE_KEY);

const supabase = useSupabase
    ? createClient(SUPABASE_URL, SUPABASE_KEY, { auth: { persistSession: false } })
    : null;

// --- Seed data -------------------------------------------------------------
const SEED_USERS = [
    { id: 'u-1', username: '1',      password: '1',        name: 'Fahmi',            type: 'intern', email: 'fahmi@calisto.com',  avatar: '', created_at: '2026-04-01T00:00:00.000Z' },
    { id: 'u-2', username: 'doctor', password: 'ocularxr', name: 'Dr. Julian Voss',  type: 'doctor', email: 'julian@calisto.com', avatar: '', created_at: '2026-04-01T00:00:00.000Z' },
    { id: 'u-3', username: 'nurse',  password: 'nurs3',    name: 'Nurse Meera Syed', type: 'nurse',  email: 'meera@calisto.com',  avatar: '', created_at: '2026-04-01T00:00:00.000Z' }
];

const SEED_RECORDS = [
    { id: 'OCU-9921', patient: 'Elena Rodriguez', age: 37, gender: 'Female', date: '2026-04-29', result: 'Healthy',              confidence: '99.2%', doctor: 'Dr. Julian Voss',  severity: 'Healthy'  },
    { id: 'OCU-9922', patient: 'Marcus Chen',     age: 41, gender: 'Male',   date: '2026-04-28', result: 'Early Glaucoma',       confidence: '87.4%', doctor: 'Dr. Julian Voss',  severity: 'Moderate' },
    { id: 'OCU-9923', patient: 'Sarah Miller',    age: 26, gender: 'Female', date: '2026-04-27', result: 'Healthy',              confidence: '98.8%', doctor: 'Dr. Julian Voss',  severity: 'Healthy'  },
    { id: 'OCU-1125', patient: 'Aliyah Rahman',   age: 59, gender: 'Female', date: '2026-04-26', result: 'Diabetic Retinopathy', confidence: '82.0%', doctor: 'Nurse Meera Syed', severity: 'Critical' },
    { id: 'OCU-1126', patient: 'Paul Garnier',    age: 72, gender: 'Male',   date: '2026-04-24', result: 'AMD',                  confidence: '91.6%', doctor: 'Dr. Julian Voss',  severity: 'Moderate' },
    { id: 'OCU-1128', patient: 'Tan Wei',         age: 48, gender: 'Male',   date: '2026-04-20', result: 'Glaucoma',             confidence: '90.8%', doctor: 'Dr. Julian Voss',  severity: 'Critical' },
    { id: 'OCU-1129', patient: 'Siti Aminah',     age: 54, gender: 'Female', date: '2026-04-18', result: 'Healthy',              confidence: '98.1%', doctor: 'Nurse Meera Syed', severity: 'Healthy'  }
];

// --- JSON file helpers (fallback only) -------------------------------------
const DATA_DIR = path.join(__dirname, 'data');
const USERS_FILE = path.join(DATA_DIR, 'users.json');
const RECORDS_FILE = path.join(DATA_DIR, 'records.json');

function ensureDataStore() {
    if (useSupabase) return;
    if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });
    if (!fs.existsSync(USERS_FILE)) fs.writeFileSync(USERS_FILE, JSON.stringify(SEED_USERS, null, 2));
    if (!fs.existsSync(RECORDS_FILE)) fs.writeFileSync(RECORDS_FILE, JSON.stringify(SEED_RECORDS, null, 2));
}

function readJsonFile(file, fallback) {
    try {
        ensureDataStore();
        const raw = fs.readFileSync(file, 'utf8');
        const parsed = JSON.parse(raw);
        return Array.isArray(parsed) ? parsed : fallback;
    } catch (err) {
        return fallback;
    }
}

function writeJsonFile(file, data) {
    ensureDataStore();
    fs.writeFileSync(file, JSON.stringify(data, null, 2));
}

// --- Supabase helpers ------------------------------------------------------
// PostgREST returns a "no rows" error code we should treat as "not found".
function unwrap(result) {
    const { data, error } = result;
    if (error && error.code !== 'PGRST116') throw error;
    return data || null;
}

// --- User operations -------------------------------------------------------
async function findUserByToken(token) {
    if (!token) return null;
    if (useSupabase) {
        return unwrap(await supabase.from('users').select('*').eq('token', token).maybeSingle());
    }
    return readJsonFile(USERS_FILE, SEED_USERS).find(u => u.token === token) || null;
}

async function findUserByUsername(username) {
    if (useSupabase) {
        return unwrap(await supabase.from('users').select('*').eq('username', username).maybeSingle());
    }
    return readJsonFile(USERS_FILE, SEED_USERS).find(u => u.username === username) || null;
}

async function findUserById(id) {
    if (useSupabase) {
        return unwrap(await supabase.from('users').select('*').eq('id', id).maybeSingle());
    }
    return readJsonFile(USERS_FILE, SEED_USERS).find(u => u.id === id) || null;
}

async function insertUser(user) {
    if (useSupabase) {
        const { data, error } = await supabase.from('users').insert(user).select().single();
        if (error) throw error;
        return data;
    }
    const users = readJsonFile(USERS_FILE, SEED_USERS);
    users.push(user);
    writeJsonFile(USERS_FILE, users);
    return user;
}

async function patchUser(id, patch) {
    if (useSupabase) {
        const { data, error } = await supabase.from('users').update(patch).eq('id', id).select().single();
        if (error) throw error;
        return data;
    }
    const users = readJsonFile(USERS_FILE, SEED_USERS);
    const idx = users.findIndex(u => u.id === id);
    if (idx === -1) return null;
    users[idx] = { ...users[idx], ...patch };
    writeJsonFile(USERS_FILE, users);
    return users[idx];
}

async function setUserToken(id, token) {
    return patchUser(id, { token });
}

// --- Record operations -----------------------------------------------------
async function getAllRecords() {
    if (useSupabase) {
        const { data, error } = await supabase
            .from('records')
            .select('*')
            .order('created_at', { ascending: false, nullsFirst: false })
            .order('date', { ascending: false });
        if (error) throw error;
        return data || [];
    }
    return readJsonFile(RECORDS_FILE, SEED_RECORDS);
}

async function findRecordById(id) {
    if (useSupabase) {
        return unwrap(await supabase.from('records').select('*').eq('id', id).maybeSingle());
    }
    return readJsonFile(RECORDS_FILE, SEED_RECORDS).find(r => r.id === id) || null;
}

async function insertRecord(record) {
    if (useSupabase) {
        const { data, error } = await supabase.from('records').insert(record).select().single();
        if (error) throw error;
        return data;
    }
    const records = readJsonFile(RECORDS_FILE, SEED_RECORDS);
    records.unshift(record);
    writeJsonFile(RECORDS_FILE, records);
    return record;
}

async function patchRecord(id, patch) {
    if (useSupabase) {
        const { data, error } = await supabase.from('records').update(patch).eq('id', id).select().single();
        if (error) throw error;
        return data;
    }
    const records = readJsonFile(RECORDS_FILE, SEED_RECORDS);
    const idx = records.findIndex(r => r.id === id);
    if (idx === -1) return null;
    records[idx] = { ...records[idx], ...patch };
    writeJsonFile(RECORDS_FILE, records);
    return records[idx];
}

async function deleteRecord(id) {
    if (useSupabase) {
        const { data, error } = await supabase.from('records').delete().eq('id', id).select().single();
        if (error && error.code !== 'PGRST116') throw error;
        return data || null;
    }
    const records = readJsonFile(RECORDS_FILE, SEED_RECORDS);
    const idx = records.findIndex(r => r.id === id);
    if (idx === -1) return null;
    const [removed] = records.splice(idx, 1);
    writeJsonFile(RECORDS_FILE, records);
    return removed;
}

// --- Bootstrap -------------------------------------------------------------
async function init() {
    if (useSupabase) {
        // Sanity check the connection. Throwing here surfaces auth / typo issues
        // at process start instead of on the first request.
        const { error } = await supabase.from('users').select('id').limit(1);
        if (error) {
            console.error('[db] Supabase connection failed:', error.message);
            throw error;
        }
        console.log('[db] storage backend: Supabase');
    } else {
        ensureDataStore();
        console.log('[db] storage backend: local JSON files (./data) — set SUPABASE_URL + SUPABASE_SERVICE_KEY to switch to Supabase');
    }
}

module.exports = {
    useSupabase,
    init,
    // users
    findUserByToken,
    findUserByUsername,
    findUserById,
    insertUser,
    patchUser,
    setUserToken,
    // records
    getAllRecords,
    findRecordById,
    insertRecord,
    patchRecord,
    deleteRecord
};
