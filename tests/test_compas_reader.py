from compas_surrogate.compas_output_parser.compas_output import CompasOutput


def test_html_repr(test_datapath):
    co = CompasOutput(test_datapath)
    html = co._repr_html_()
    assert isinstance(html, str)
