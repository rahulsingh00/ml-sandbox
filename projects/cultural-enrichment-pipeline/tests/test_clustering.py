"""
Unit tests for clustering.py using pytest.
Tests the CulturalTrendClusterer.
"""

from clustering import CulturalTrendClusterer


def test_fit_and_predict_basic():
    clusterer = CulturalTrendClusterer(num_clusters=2)
    documents = [
        "Tech startup launches new computer hardware next week.",
        "Silicon Valley startup builds advanced software pipeline.",
        "Heavy rain and storms forecasted for the East Coast.",
        "Hurricane alerts warning citizens to seek shelter."
    ]
    
    assignments = clusterer.fit_and_predict(documents)
    
    assert len(assignments) == 4
    # Check that predictions are valid cluster IDs
    assert all(a in [0, 1] for a in assignments)
    
    keywords = clusterer.get_cluster_keywords()
    assert len(keywords) == 2
    for cid, kws in keywords.items():
        assert len(kws) <= 5
        assert all(isinstance(kw, str) for kw in kws)


def test_edge_cases():
    clusterer = CulturalTrendClusterer(num_clusters=1)
    documents = [
        "Single document test sample."
    ]
    assignments = clusterer.fit_and_predict(documents)
    assert assignments == [0]
    
    keywords = clusterer.get_cluster_keywords()
    assert len(keywords) == 1
