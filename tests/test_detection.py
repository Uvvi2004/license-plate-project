from license_plate_pipeline.detection import pad_box


def test_pad_box_pads_each_side():
    x1, y1, x2, y2 = pad_box(100, 100, 200, 150, img_w=1000, img_h=1000, pad_ratio=0.1)
    assert (x1, y1, x2, y2) == (90, 95, 210, 155)


def test_pad_box_clamps_to_image_bounds():
    x1, y1, x2, y2 = pad_box(0, 0, 20, 20, img_w=1000, img_h=1000, pad_ratio=0.5)
    assert x1 == 0
    assert y1 == 0

    x1, y1, x2, y2 = pad_box(980, 980, 1000, 1000, img_w=1000, img_h=1000, pad_ratio=0.5)
    assert x2 == 1000
    assert y2 == 1000
