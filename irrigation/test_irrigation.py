import unittest
from irrigation_controller import IrrigationController

class TestIrrigationController(unittest.TestCase):
    def setUp(self):
        self.controller = IrrigationController(30, 40)

    def test_dry_turns_pump_on(self):
        s = self.controller.update(25)
        self.assertTrue(s.pump_on); self.assertEqual(s.status, "DRY")

    def test_wet_turns_pump_off(self):
        self.controller.update(25)
        s = self.controller.update(45)
        self.assertFalse(s.pump_on); self.assertEqual(s.status, "ADEQUATE")

    def test_hysteresis_keeps_pump_on(self):
        self.controller.update(25)
        s = self.controller.update(35)
        self.assertTrue(s.pump_on); self.assertEqual(s.status, "WATERING")

    def test_hysteresis_keeps_pump_off(self):
        s = self.controller.update(35)
        self.assertFalse(s.pump_on); self.assertEqual(s.status, "STABLE")

    def test_invalid_moisture(self):
        with self.assertRaises(ValueError): self.controller.update(101)

    def test_invalid_thresholds(self):
        with self.assertRaises(ValueError): IrrigationController(50, 40)

if __name__ == "__main__": unittest.main()
