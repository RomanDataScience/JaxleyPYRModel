import pytest

from jaxley_refactored.distributions import combe2023_distributions
from jaxley_refactored.parameters import combe2023_catalog


def test_distribution_registry_validates_and_canonicalizes_coefficients():
    catalog = combe2023_catalog()
    registry = combe2023_distributions(catalog)

    assert registry.resolve(
        "combe2023_cch_driven", {"DistHalfRm": 175.0}
    ) == {"DistHalfRm": 175.0}
    with pytest.raises(ValueError, match="outside"):
        registry.resolve("combe2023_cch_driven", {"DistHalfRm": 10_000.0})
    with pytest.raises(KeyError, match="Unknown distribution profile"):
        registry.resolve("other", {})
