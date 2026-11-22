import unittest

from homr.model import MultiStaff, Staff, StaffPoint
from homr.staff_regions import StaffRegions


def build_staff(y_min: int, y_max: int) -> Staff:
    step = (y_max - y_min) // 4
    p1 = StaffPoint(0, list(range(y_min, y_max + 1, step)), 0)
    p2 = StaffPoint(100, list(range(y_min, y_max + 1, step)), 0)
    return Staff([p1, p2])


class TestStaffRegions(unittest.TestCase):
    def test_staff_above(self) -> None:
        system1 = MultiStaff(
            [
                build_staff(100, 200),
                build_staff(300, 400),
            ],
            [],
        )
        system2 = MultiStaff(
            [
                build_staff(500, 600),
                build_staff(700, 800),
            ],
            [],
        )
        regions = StaffRegions([system1, system2])
        self.assertEqual(0, regions.get_start_of_closest_staff_above(100))
        self.assertEqual(200, regions.get_start_of_closest_staff_above(201))
        self.assertEqual(400, regions.get_start_of_closest_staff_above(500))
        self.assertEqual(600, regions.get_start_of_closest_staff_above(633))
        self.assertEqual(800, regions.get_start_of_closest_staff_above(1000))

    def test_staff_below(self) -> None:
        system1 = MultiStaff(
            [
                build_staff(100, 200),
                build_staff(300, 400),
            ],
            [],
        )
        system2 = MultiStaff(
            [
                build_staff(500, 600),
                build_staff(700, 800),
            ],
            [],
        )
        regions = StaffRegions([system1, system2])
        self.assertEqual(100, regions.get_start_of_closest_staff_below(0))
        self.assertEqual(100, regions.get_start_of_closest_staff_below(99))
        self.assertEqual(700, regions.get_start_of_closest_staff_below(600))
        self.assertEqual(1e12, regions.get_start_of_closest_staff_below(800))


class TestCalculateRegion(unittest.TestCase):
    def test_overlapping_staffs_keep_a_positive_region(self) -> None:
        # A second detection overlapping the staff makes the neighbour limits cross; the region
        # used to come out upside down and crash the canvas resize for the whole page.
        from homr.staff_parsing import _calculate_region

        staff = build_staff(300, 400)
        overlapping = build_staff(250, 350)
        regions = StaffRegions([MultiStaff([staff], []), MultiStaff([overlapping], [])])
        x_min, y_min, x_max, y_max = _calculate_region(staff, regions)
        self.assertGreater(y_max - y_min, 100)
        self.assertLessEqual(y_min, 300)
        self.assertGreaterEqual(y_max, 400)

    def test_separate_staffs_use_neighbour_limits(self) -> None:
        from homr.staff_parsing import _calculate_region

        staff = build_staff(300, 400)
        regions = StaffRegions(
            [MultiStaff([build_staff(100, 200)], []), MultiStaff([staff], []),
             MultiStaff([build_staff(500, 600)], [])]
        )
        _, y_min, _, y_max = _calculate_region(staff, regions)
        self.assertEqual((y_min, y_max), (200, 500))
