# Notes on this project

## Experimental

### RTD Calibration

The RTD was calibrated and the 3rd order calibration coefficients can be found in the file [coefficients.csv](resources/experimental-set-calibration/Processing/coefficients.csv).

### Wire temperature measurement

- Thermocouple tying: DISCARDED.
  - what is: the thermocouple legs were tied to the wire without welding. The main advantage of this method is avoiding electrical contact as happens in a welding.
  - problems:
    - bubbles began to form in the tie much earlier than on the rest of the wire.
    - the thermocouple hardly measured any increase in the temperature even with a high increase in the current.
  - possible reasons: the earliest contact point between the thermocouple legs (effectively the measuring point) did not touch the wire surface. It actually happened above the surface, causing the sensor to measure the temperature of the water only.

- Capacitive discharge: DISCARDED.
  - what is: the thermocouple legs are welded on the wire's surface using capacitive discharge. For each leg, the leg is connected to one pole of a capacitor, while the wire is connected to the other pole. When the wire and the leg touch, the capacitor discharges and the leg is welded on the wire. This procedure is repeated for both thermocouple's legs.
  - observations:
    - for some of the tested wires, temperature is extremely sensitive to variations in the applied current. For instance, ~900°C variations in the measurements were easily achieved.
    - on the other hand, some wires "measured" a *decrease* in the temperature with an increase in the current. One measurement showed ~ -400°C, which obviously doesn't make sense.
    - an interesting observation is that, before applying current, the temperature measured by the thermocouple matches almost perfectly the RTD's measurements. That means that the problem appears when applying current only, and suggests that the problem is of electrical nature.
  - problems:
    - the welding of the thermocouple creates a new material that should be calibrated. However, by inserting the thermocouple together with the wire into the calibration chamber, the whole system's temperature will be uniform, and ir will behave just like a perfect, undamaged thermocouple. On the other hand, during the experiment, there is a strong temperature gradient through the resistance-welding point-thermocouple wires system, leading to high errors.
    - because the two legs of the thermocouple cannot be welded for sure at the same axial position, there are two factors for voltage drop between them: the first one, the one we want to measure, is a voltage drop caused by the temperature difference between our system and the cold joint; the second one is a voltage drop due to the current that flows through the test wire, amounting to over 900°C for a distance of 1mm between the thermocouple legs.

- Ceramic capsule: DISCARDED.
  - what is: a small, cylindrical ceramic capsule with two orifices, each going from one base of the cylinder to the other. The wire is inserted into one orifice, whereas the welded thermocouple was inserted into the other.
  - problems: the thermocouple does not capture the wire temperature.
  - observations:
    - bubbles begin to form inside the capsule, and much earlier than expected or on the remainder of the wire.
    - as heat flux is increased, the measured wire temperature increases until a certain point. Thereafter it remains virtually constant, whereas we know that it should keep increasing.
  - possible reasons: the capsule acts as a cavity in which water vapor is more easily trapped, and thus favouring boiling.

- Silver-welding: DISCARDED.
  - what is: the thermocouple is welded at the wire surface using silver as intermediate material.
  - problems:
    - the base material is damaged due to the high temperatures.
    - there is a big fin, the welding point, which dissipates heat much faster than on the rest of the wire. This is especially problematic when we take into account that it is exactly at the welding point that temperature is sensed by the thermocouple.
  - observations: measured temperature does not rise as steeply as expected. In fact, measured temperature is very resistant to increases in the heat flux. This does not agree with what we expected.

- Glueing: IN PROGRESS.
  - variations:
    - big glue drop:
      - what is: in this method, the thermocouple legs are previously welded one to another, and the welding point is subsequently glued onto the wire's surface with a special glue.
      - problems: as observed in the other methods, the heat transfer behavior is visibly different at the connection. At the point where the thermocouple is glued onto the wire, bubble formation rate is higher and the measured temperature does not vary as expected.
    - glue + isolating tape:
      - what is: in this method, the thermocouple joint is glued onto the wire's surface, and this joint is covered with an isolating tape.
      - observations: bubble formation started much earlier on the tape. Also, the critical point was achieved with a much lower heat flux, and the wire broke inside the covered region.
      - problem: the observations demonstrate that the tape strongly modifies the heat transfer behavior of the heating wire.
    - thin layers:
      <!-- TODO: here -->

## Sample

### Wire 1

Wire 1 was used in Experiment 1. Data:

- sample id: #1
- code: NI80-020-200
- diameter: 0.51mm (large)
- length: 6.5cm
- texturing: none. The wire was used as extracted from the coil. No further treatment was applied.

In the experiment, there as a T-type thermocouple glued to the wire surface.

### Wire 2

Wire 2 was used in Experiment 2. Data:

- sample id: #2
- code: NI80-020-200
- diameter: 0.51mm (large)
- length: 6.5cm
- texturing: none. The wire was used as extracted from the coil. No further treatment was applied.

The experiment was executed without sensors.

### Wire 3

Wire 3 was used in Experiment 3. Data:

- sample id: #3
- code: NI80-010-200
- diameter: 0.25mm (small)
- length: 6.5cm
- texturing: none. The wire was used as extracted from the coil. No further treatment was applied.

The experiment was executed without sensors.

### Horizontal Ribbon

Ribbon 1 was used in Experiment 4. Data:

- sample id: #4
- code: NCRR-17-100
- width: 1/16in
- thickness: 0.0031in
- length: 6.5cm
- texturing: none. The ribbon was used as extracted from the coil. No further treatment was applied.

The experiment was executed without sensors.

### Vertical Ribbon

Ribbon 2 was used in Experiment 5. Data:

- sample id: #5
- code: NCRR-17-100
- width: 1/16in
- thickness: 0.0031in
- length: 6.5cm
- texturing: none. The ribbon was used as extracted from the coil. No further treatment was applied.

The experiment was executed without sensors.

## Machine learning

### Bubble detection

I searched for some papers that used machine learning or filters to detect bubbles. However, the best studies I found showed that CNNs are the way to go, having a much better accuracy in detecting bubbles and avoiding false positives. Perhaps more research is needed before deciding to exclude other methods and diving into CNNs. <!-- TODO: more research on different bubble detection methods -->
