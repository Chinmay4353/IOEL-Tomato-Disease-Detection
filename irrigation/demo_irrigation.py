from irrigation_controller import IrrigationController

controller = IrrigationController(30, 40)
readings = [45, 35, 29, 25, 31, 37, 40, 45]
for moisture in readings:
    s = controller.update(moisture)
    print(f"Moisture={s.moisture_percent:5.1f}% | Status={s.status:9s} | Pump={'ON' if s.pump_on else 'OFF'}")

