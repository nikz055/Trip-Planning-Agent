-- Trip-planning agent: initial schema.
--
-- Access model: the browser never talks to the database. Every read and write
-- goes through the backend, which connects as the database owner. Row Level
-- Security is enabled on every table with no policies, and table privileges
-- are revoked from the API roles, so the public (anon / publishable) key can
-- read and write nothing.
--
-- No table stores passport, ID or payment data.

create table public.trips (
  id                 uuid primary key default gen_random_uuid(),  -- also the share link
  state              text not null default 'NEW'
    check (state in ('NEW','WAITING_FOR_DETAILS','PLANNING','VALIDATING','REPAIRING',
                     'PRESENTED','REVISING','ACCEPTED','PARTIAL','FAILED','CANCELLED')),
  request            jsonb not null,
  current_version    integer,
  locked_item_ids    text[] not null default '{}',
  iteration_count    integer not null default 0 check (iteration_count >= 0),
  tokens_used        bigint  not null default 0 check (tokens_used >= 0),
  estimated_cost_usd numeric(10,4) not null default 0 check (estimated_cost_usd >= 0),
  repair_used        boolean not null default false,
  pause              jsonb,                               -- pause reason: the form being waited on
  runtime            jsonb not null default '{}'::jsonb,  -- run bookkeeping: unavailable tools, pending revision, notices, stats
  -- The run lease. At most one holder at a time; it expires at run_deadline so a
  -- crashed run cannot hold the trip forever.
  run_id             text,
  run_deadline       timestamptz,
  created_at         timestamptz not null default now(),
  updated_at         timestamptz not null default now(),
  check ((run_id is null) = (run_deadline is null))
);
create index trips_state_idx on public.trips (state);

create table public.plan_versions (
  trip_id        uuid not null references public.trips (id) on delete cascade,
  version        integer not null check (version >= 1),
  reason         text not null check (reason in ('initial','repair','revision','refresh')),
  parent_version integer,
  plan           jsonb not null,
  created_at     timestamptz not null default now(),
  primary key (trip_id, version),  -- also indexes the trip_id foreign key
  check (parent_version is null or parent_version < version),
  foreign key (trip_id, parent_version) references public.plan_versions (trip_id, version)
);

-- The trip points at one of its own versions. Deferred so the version and the
-- pointer can be written in one transaction in either order.
alter table public.trips
  add constraint trips_current_version_fk
  foreign key (id, current_version) references public.plan_versions (trip_id, version)
  deferrable initially deferred;

create table public.tool_calls (
  trip_id     uuid not null references public.trips (id) on delete cascade,
  call_ref    text not null,                 -- short id the planner cites, e.g. tc_0007
  tool        text not null,                 -- no CHECK: rejected calls to unknown tools are recorded too
  args        jsonb not null,
  depends_on  text[] not null default '{}',
  status      text not null check (status in ('pending','running','ok','failed')),
  attempts    integer not null default 0 check (attempts >= 0),
  label       text,
  llm_call_id text,
  error       text,
  iteration   integer not null default 0,
  started_at  timestamptz,
  finished_at timestamptz,
  created_at  timestamptz not null default now(),
  primary key (trip_id, call_ref)            -- also indexes the trip_id foreign key
);

create table public.tool_results (
  id            uuid primary key default gen_random_uuid(),
  trip_id       uuid not null,
  call_ref      text not null,
  payload       jsonb,
  fetched_at    timestamptz not null,        -- a stored price is a record of a search, not a live offer
  source        text not null,
  error         text,
  superseded_at timestamptz,                 -- set when BR-10 re-fetches this call
  foreign key (trip_id, call_ref) references public.tool_calls (trip_id, call_ref) on delete cascade
);
create index tool_results_call_idx on public.tool_results (trip_id, call_ref);
create unique index tool_results_one_current
  on public.tool_results (trip_id, call_ref) where superseded_at is null;

create table public.trip_messages (
  trip_id    uuid not null references public.trips (id) on delete cascade,
  seq        integer not null check (seq >= 0),
  message    jsonb not null,
  created_at timestamptz not null default now(),
  primary key (trip_id, seq)                 -- also indexes the trip_id foreign key
);

create table public.user_actions (
  id              uuid primary key default gen_random_uuid(),
  trip_id         uuid not null references public.trips (id) on delete cascade,
  idempotency_key text not null check (length(idempotency_key) between 8 and 200),
  kind            text not null check (kind in ('create','submit_form','lock','revise','accept','cancel')),
  received_at     timestamptz not null default now(),
  result          jsonb,                     -- null while the action is still running
  unique (trip_id, idempotency_key)          -- also indexes the trip_id foreign key
);
-- A repeated "create" has no trip id to look up, so its key is unique on its own.
create unique index user_actions_create_key
  on public.user_actions (idempotency_key) where kind = 'create';

-- Row Level Security on every table, with no policies.
alter table public.trips         enable row level security;
alter table public.plan_versions enable row level security;
alter table public.tool_calls    enable row level security;
alter table public.tool_results  enable row level security;
alter table public.trip_messages enable row level security;
alter table public.user_actions  enable row level security;

revoke all on public.trips, public.plan_versions, public.tool_calls,
              public.tool_results, public.trip_messages, public.user_actions
  from anon, authenticated;
