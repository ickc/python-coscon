from coscon.toast_helper import fake_focalplane, FakeFocalPlane


def test_fake_focalplane():
    fp = fake_focalplane()
    fp_ = FakeFocalPlane(fp)
    df = fp_.dataframe
