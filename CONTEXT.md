# QS

Automated visual inspection of моти before packing, advising an operator whether each unit may go on.

## Language

**Unit**:
The one thing a verdict belongs to. On the line it is one cup with whatever is in it; on the bench, where there are no cups, it is one моти.
_Avoid_: Item, piece, product

**Cup**:
The clear container that carries a моти along the chain.
_Avoid_: Stakan, tray

**Моти**:
The product itself: a soft rice-dough round, dusted with starch, with a filling.
_Avoid_: Mochi, unit (on the line)

**Empty cup**:
A cup with no моти in it. It is a defective unit, because a line station sits after the last filling step, right before packing; upstream of that point an empty cup may still be filled and the station is mounted in the wrong place.
_Avoid_: Missing unit, no unit

**Gap**:
The stretch of chain between two cups. It is not a unit and gets no verdict.
_Avoid_: No cup, nothing, empty slot

**Trigger**:
The signal that a unit has arrived and must be inspected.
_Avoid_: Event, detection

**Missed cup**:
A cup that passes the camera with no trigger, so it gets no verdict.
_Avoid_: Skipped unit, lost unit

**Check**:
One pass-or-fail test of a unit. ГОДЕН needs every configured check to have run and passed.
_Avoid_: Test, rule

**Empty-cup check**:
The check that a found cup holds a моти. A trained model that has seen only full cups judges it: a cup that looks unlike them fails.
_Avoid_: Moti detector

**Warmth guard**:
A measured colour check that runs beside the empty-cup check. It can withhold ГОДЕН, turning it into НЕТ РЕШЕНИЯ, but it never passes a cup by itself.
_Avoid_: Warmth verdict, second check

**Empty-cup catch rate**:
The share of empty cups that get БРАК. The client's 95% is this rate, measured at the false-alarm cap.
_Avoid_: Accuracy, recall (without saying of what)

**False-alarm cap**:
The highest share of full cups allowed to get БРАК. A catch rate means nothing without it.
_Avoid_: Precision, error rate

**Inspection zone**:
The fixed part of the view where a unit is judged. A cup is judged on several frames that each show all of it inside the zone; a cup seen in view that never gets enough such frames ends in НЕТ РЕШЕНИЯ.
_Avoid_: Inspection area, frame, placement area

**Verdict**:
The station's single answer about one unit: ГОДЕН, БРАК or НЕТ РЕШЕНИЯ.
_Avoid_: Result, decision
