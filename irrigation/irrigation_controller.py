from dataclasses import dataclass

@dataclass
class IrrigationState:
    moisture_percent: float
    pump_on: bool
    status: str

class IrrigationController:
    def __init__(self, dry_threshold=30.0, wet_threshold=40.0):
        if not 0 <= dry_threshold < wet_threshold <= 100:
            raise ValueError("Thresholds must satisfy 0 <= dry_threshold < wet_threshold <= 100.")
        self.dry_threshold = dry_threshold
        self.wet_threshold = wet_threshold
        self.pump_on = False

    def update(self, moisture_percent):
        if not 0 <= moisture_percent <= 100:
            raise ValueError("moisture_percent must be between 0 and 100.")
        if moisture_percent <= self.dry_threshold:
            self.pump_on = True
            status = "DRY"
        elif moisture_percent >= self.wet_threshold:
            self.pump_on = False
            status = "ADEQUATE"
        else:
            status = "WATERING" if self.pump_on else "STABLE"
        return IrrigationState(moisture_percent, self.pump_on, status)
