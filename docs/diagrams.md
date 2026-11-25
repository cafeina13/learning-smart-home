# Design diagrams

Mermaid diagrams for the report. They render on GitHub and in most Markdown
viewers. `scripts/render_diagrams.py` turns each block into a PNG in
`docs/figures/` (Mermaid in a headless Edge or Chrome, no install).

## 1. System architecture

Everything talks over one local MQTT broker. The article's devices and the
household's appliances are simulated; a real device would publish on the same
topics.

```mermaid
flowchart TB
    subgraph Home["Simulated home"]
        S["Article's<br/>sensors,<br/>actuators"]
        P["Smart-plug<br/>appliances"]
        M["Delay-start<br/>machines"]
    end
    B(("MQTT<br/>broker"))
    subgraph Logic["Home logic"]
        R["Rules<br/>(3 strikes<br/>= off)"]
        SS["Smart<br/>start"]
        F["Forest<br/>guard"]
        L[("Event<br/>log")]
    end
    subgraph Learn["Habit learning"]
        H["Counting<br/>routines"]
        FC["Habit<br/>forest"]
    end
    D["Dashboard:<br/>accept or<br/>reject"]
    U(("Person"))
    S & P & M <--> B
    B <--> R & SS & F & L
    L --> H & FC
    H & FC --> D
    U <--> D
```

## 2. Topics and messages

```mermaid
flowchart LR
    P1["State payload<br/>device, kind,<br/>value, ts,<br/>source, extra"] --- T1["home/room/device/state<br/>a device reports<br/>its value"]
    P2["Command payload<br/>value, ts, source<br/>(user, dashboard,<br/>rule, scheduler,<br/>forest)"] --- T2["home/room/device/set<br/>someone asks a<br/>device to change"]
    T3["home/alerts<br/>alerts, refusals,<br/>rules off, rescues"]
    T4["home/plans/room/device<br/>smart-start options"]
```

## 3. What a rule may do to each kind of device

People can always operate a device. Commands from rules and the scheduler are
checked by the device itself.

```mermaid
stateDiagram-v2
    direction LR
    state "Manual appliance behind a smart plug (TV, computer)" as Manual {
        [*] --> off
        off --> on: person switches on
        on --> off: person switches off
        off --> cut: rule cuts power
        on --> on: rule cuts power (waits)
        on --> cut: person switches off<br/>while a cut waits
        cut --> off: rule or forest<br/>restores power
        cut --> on: person switches on<br/>= OVERRIDE (counted)
    }
```

```mermaid
stateDiagram-v2
    direction LR
    state "Armable machine (dishwasher, washing machine)" as Armable {
        [*] --> off
        off --> armed: person loads it and sets a deadline
        armed --> on: scheduler starts it<br/>(only while armed)
        off --> on: person starts it now
        armed --> off: person cancels
        on --> off: cycle ends
    }
```

## 4. Smart start (hybrid design)

```mermaid
sequenceDiagram
    actor P as Person
    participant D as Dashboard
    participant W as Dishwasher
    participant S as Scheduler
    P->>D: loaded,<br/>finish by 18:00
    D->>W: armed,<br/>deadline 18:00
    W-->>S: armed +<br/>deadline
    S->>S: score each start<br/>(15-min steps) by<br/>solar left after<br/>usual house load
    S-->>D: 3 options,<br/>best preselected
    opt another time
        P->>D: pick option 2
        D->>S: choose(start)
    end
    Note over S: chosen time
    S->>W: on (source:<br/>scheduler)
    W-->>D: on, then off<br/>at cycle end
```

## 5. Learning from being wrong

A strike is a person's override, or a cut the habit forest ended early because
it expected a use (the rule was about to be wrong).

```mermaid
sequenceDiagram
    participant R as Rule engine
    participant TV as TV plug
    participant F as Forest
    actor P as Person
    R->>TV: cut (quiet<br/>window starts)
    alt use likely this hour
        F->>TV: restore
        F-->>R: rescued
        R->>R: strike
    else nobody foresaw it
        P->>TV: switch on
        TV-->>R: override
        R->>R: strike
    end
    Note over R: more strikes<br/>on later days
    R->>R: 3 strikes:<br/>rule off
    R-->>P: alert
```

## 6. Data and evaluation pipeline

Every reported number comes from one `eval_*` script, run on the prepared
tables of one dataset (`CAPSTONE_DATASET` = refit, ukdale or casas).

```mermaid
flowchart LR
    R1[("REFIT<br/>20 homes")]
    R2[("UK-DALE<br/>5 homes")]
    R3[("CASAS<br/>21 homes")]
    T[("Same tables<br/>for all three:<br/>uses, hours,<br/>appliances")]
    PV[("PVGIS<br/>solar")]
    subgraph EXP["One script per result"]
        E1["eval_refit,<br/>eval_habit_networks:<br/>habit models"]
        E2["eval_houses,<br/>eval_external"]
        E3["eval_suggestions<br/>cuts, smart start"]
        E4["eval_estimates"]
        E5["eval_forecast,<br/>eval_forecast_habits"]
        E6["eval_gad"]
        E7["eval_split_effect"]
    end
    R1 -- prepare_refit --> T
    R2 -- prepare_ukdale --> T
    R3 -- prepare_casas --> T
    T --> EXP
    PV --> E3 & E4 & E5
```

## 7. The habit forest in the loop

Counting proposes, the forest checks and guards (`habits/forest.py`).

```mermaid
flowchart TB
    subgraph GUARD["Step 2: guard"]
        direction TB
        CUT["Accepted cut<br/>is active"] --> HOUR{"Each hour:<br/>use likely?"}
        HOUR -- no --> CUT
        HOUR -- yes --> RES["Restore power<br/>early"]
        RES --> STR["Strike for<br/>the rule"]
    end
    subgraph CHECK["Step 1: check"]
        direction TB
        U[("Past uses")] --> C["Counting<br/>finds a window"]
        U --> FIT["Forest: train on<br/>first 60% of days,<br/>threshold on next 20%"]
        C --> Q{"Agrees on<br/>half of the<br/>last 20%?"}
        FIT --> Q
        Q -- no --> HOLD["Held back"]
        Q -- yes --> DASH["Shown with<br/>its agreement"]
    end
```
