from coscon.toast_helper import FakeFocalPlane, fake_focalplane


def test_fake_focalplane():
    fp = fake_focalplane()
    fp_ = FakeFocalPlane(fp)
    df = fp_.dataframe
