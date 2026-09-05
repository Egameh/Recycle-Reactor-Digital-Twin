# Tennessee Eastman Process (TEP) — Plant Overview

## What the Plant Does

The Tennessee Eastman Process is a simulated chemical plant that turns four gas
reactants (called A, C, D, and E) into two liquid products (called G and H),
along with a small amount of an unwanted liquid byproduct (called F). An inert
gas (called B) also passes through the system without reacting — it just needs
to be vented out periodically so it doesn't build up.

Think of it as: raw gases go in one end, useful liquid products come out the
other end, and along the way the plant has to separate out what it doesn't
want and recycle anything reusable back into the process.

## The Five Main Pieces of Equipment

The plant is built from five connected units, and material flows through them
in roughly this order:

1. **Reactor** — This is where the actual chemistry happens. The gas
   reactants are combined here, and a chemical reaction converts them into
   the liquid products. The reaction gives off heat (it's exothermic), so
   the reactor needs cooling to keep from overheating.

2. **Condenser** — The output from the reactor is a mix of gas and liquid.
   The condenser cools this mixture down so more of it turns into liquid,
   making it easier to separate afterward.

3. **Separator** — This unit splits the stream into a liquid portion (which
   continues on toward the final product) and a vapor portion (gas that
   didn't condense — mostly unreacted material).

4. **Compressor** — The uncondensed vapor from the separator isn't wasted.
   The compressor pressurizes it and sends it back to the reactor as a
   recycle stream, so unreacted material gets another chance to react
   instead of being thrown away. A small purge stream is vented off here too,
   to prevent the inert gas and byproduct from slowly building up in the
   recycle loop over time.

5. **Stripper** — The liquid from the separator still has some lighter,
   unwanted components dissolved in it. The stripper uses steam to boil
   those off, leaving behind the final, purified liquid product.

## Why This Plant Is Considered "Hard" to Control

Two things make TEP a genuinely challenging control problem, not just a
simple flow-through system:

- **It's open-loop unstable.** Left alone with no automatic control, the
  plant's conditions (like reactor pressure and level) won't just settle
  down on their own — they'll drift. This means the plant absolutely
  depends on active, continuous automatic control to stay in a safe,
  steady operating state.

- **Everything is connected via the recycle loop.** Because unreacted
  material gets sent back to the reactor, a disturbance introduced
  anywhere in the plant doesn't stay local — it can travel through the
  recycle loop and show up as an effect somewhere else entirely. This is
  why, for example, a temperature disturbance introduced at the reactor
  might end up being most visible in the separator or stripper pressure
  readings instead.

## What "Faults" Mean in This Simulation

The simulation includes 21 pre-programmed disturbances (numbered 1 through
21) that can be introduced to test how well a control system — or a
monitoring/diagnostic system — responds. These aren't random glitches; each
one represents a specific, realistic thing that can go wrong in a real
chemical plant, such as:

- A feed stream's composition or temperature suddenly shifting (a "step"
  disturbance)
- A feed stream becoming noisy or randomly variable rather than steady
- A control valve physically sticking instead of responding smoothly
- The underlying reaction kinetics slowly drifting over time
- A few disturbances are deliberately left undocumented ("unknown") to test
  whether a monitoring system can flag something is wrong even without
  knowing in advance what specifically caused it

## Why Some Faults Look "Worse" Than Others

Not every fault produces a big, obvious signal. The plant's own automatic
control loops are actively working to counteract disturbances in real time.
If a control loop is well-tuned for a particular type of disturbance, it may
absorb the disturbance almost completely — the plant keeps running close to
normal, and only a subtle, hard-to-spot signature remains. Other
disturbances — especially ones that remove or block a feed entirely, or
otherwise overwhelm what the control loops can correct for — will produce a
large, unmistakable deviation from normal operating conditions.

This is an important thing to understand when interpreting monitoring or
diagnostic results: a "small" detected deviation doesn't necessarily mean
the underlying disturbance is small — it may just mean the plant's control
system is successfully doing its job.

## The Variables at a Glance

- **Manipulated variables (XMV)** — These are the things an operator or
  control system can actively adjust, like valve positions and flow rates
  (for example, the reactor cooling water flow, or the purge valve
  position).
- **Measured variables (XMEAS)** — These are the readings the plant reports,
  like pressures, temperatures, levels, and flow rates, plus periodic
  composition analyses of the feed, purge gas, and final product streams.

Together, these 52 variables (41 measured + 11 manipulated) are what any
monitoring, fault-detection, or digital-twin system — including this
project's — uses to understand what's happening inside the plant at any
given moment.

---
*This overview paraphrases and summarizes publicly available technical
descriptions of the TEP benchmark (Downs & Vogel, 1993, and subsequent
open-access literature) for accessible reference. For precise variable
definitions, consult the technical variable tables in this project's
repository.*
