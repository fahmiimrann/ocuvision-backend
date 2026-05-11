-- OcuVision Supabase setup.
-- Run this entire file once in: Supabase dashboard -> SQL Editor -> New query.
-- It is safe to re-run; everything uses IF NOT EXISTS / ON CONFLICT DO NOTHING.

-- ---------------------------------------------------------------------------
-- USERS
-- ---------------------------------------------------------------------------
create table if not exists public.users (
    id          text primary key,
    username    text unique not null,
    password    text not null,
    name        text not null,
    type        text not null,
    email       text default '',
    avatar      text default '',
    token       text,
    created_at  timestamptz default now()
);

create index if not exists idx_users_token on public.users (token);

-- ---------------------------------------------------------------------------
-- PATIENT RECORDS
-- ---------------------------------------------------------------------------
create table if not exists public.records (
    id            text primary key,
    patient       text not null,
    age           integer default 0,
    gender        text default 'Other',
    date          date,
    result        text default '',
    confidence    text default '0%',
    doctor        text default 'Unassigned',
    severity      text default 'Healthy',
    fundus_image  text,
    created_by    text,
    updated_by    text,
    created_at    timestamptz default now(),
    updated_at    timestamptz
);

-- For databases that were created before the fundus_image column existed.
alter table public.records add column if not exists fundus_image text;

create index if not exists idx_records_created_at on public.records (created_at desc);

-- ---------------------------------------------------------------------------
-- ROW LEVEL SECURITY
-- We only access these tables from the Node backend using the
-- SERVICE_ROLE key, which bypasses RLS. We still enable RLS so anyone using
-- the anon (public) key cannot read or write directly.
-- ---------------------------------------------------------------------------
alter table public.users   enable row level security;
alter table public.records enable row level security;

-- ---------------------------------------------------------------------------
-- SEED USERS  (passwords are intentionally simple for the demo)
-- ---------------------------------------------------------------------------
insert into public.users (id, username, password, name, type, email) values
    ('u-1', '1',      '1',        'Fahmi',            'intern', 'fahmi@calisto.com'),
    ('u-2', 'doctor', 'ocularxr', 'Dr. Julian Voss',  'doctor', 'julian@calisto.com'),
    ('u-3', 'nurse',  'nurs3',    'Nurse Meera Syed', 'nurse',  'meera@calisto.com')
on conflict (username) do nothing;

-- ---------------------------------------------------------------------------
-- SEED RECORDS
-- ---------------------------------------------------------------------------
insert into public.records (id, patient, age, gender, date, result, confidence, doctor, severity) values
    ('OCU-9921', 'Elena Rodriguez', 37, 'Female', '2026-04-29', 'Healthy',              '99.2%', 'Dr. Julian Voss',  'Healthy'),
    ('OCU-9922', 'Marcus Chen',     41, 'Male',   '2026-04-28', 'Early Glaucoma',       '87.4%', 'Dr. Julian Voss',  'Moderate'),
    ('OCU-9923', 'Sarah Miller',    26, 'Female', '2026-04-27', 'Healthy',              '98.8%', 'Dr. Julian Voss',  'Healthy'),
    ('OCU-1125', 'Aliyah Rahman',   59, 'Female', '2026-04-26', 'Diabetic Retinopathy', '82.0%', 'Nurse Meera Syed', 'Critical'),
    ('OCU-1126', 'Paul Garnier',    72, 'Male',   '2026-04-24', 'AMD',                  '91.6%', 'Dr. Julian Voss',  'Moderate'),
    ('OCU-1128', 'Tan Wei',         48, 'Male',   '2026-04-20', 'Glaucoma',             '90.8%', 'Dr. Julian Voss',  'Critical'),
    ('OCU-1129', 'Siti Aminah',     54, 'Female', '2026-04-18', 'Healthy',              '98.1%', 'Nurse Meera Syed', 'Healthy')
on conflict (id) do nothing;
