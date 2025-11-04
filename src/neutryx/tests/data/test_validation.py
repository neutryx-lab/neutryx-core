from datetime import datetime, timedelta

from neutryx.data.validation import (
    DataValidator,
    RangeRule,
    RequiredFieldRule,
    Severity,
    StalenessRule,
)
from neutryx.market.data_models import DataQuality, FXQuote


def _make_fx_quote(spot: float = 1.10) -> FXQuote:
    return FXQuote(
        timestamp=datetime.utcnow(),
        source="simulator",
        quality=DataQuality.REALTIME,
        currency_pair="EUR/USD",
        base_currency="EUR",
        quote_currency="USD",
        spot=spot,
        bid=spot - 0.001,
        ask=spot + 0.001,
    )


def test_validator_accepts_clean_data():
    validator = DataValidator(
        [
            RequiredFieldRule(["timestamp"], severity=Severity.ERROR),
            RangeRule("spot", minimum=0.0, severity=Severity.ERROR),
        ]
    )

    quote = _make_fx_quote()
    result = validator.validate(quote)

    assert result.is_valid
    assert quote.quality == DataQuality.REALTIME
    assert not result.issues


def test_validator_flags_negative_spot():
    validator = DataValidator([RangeRule("spot", minimum=0.0, severity=Severity.ERROR)])
    quote = _make_fx_quote(spot=-0.5)

    result = validator.validate(quote)

    assert not result.is_valid
    assert quote.quality == DataQuality.STALE
    assert result.issues
    assert result.issues[0].severity == Severity.ERROR


def test_validator_marks_stale_data():
    validator = DataValidator([StalenessRule(timedelta(seconds=1))])

    quote = _make_fx_quote()
    quote.timestamp = datetime.utcnow() - timedelta(seconds=5)

    result = validator.validate(quote)

    assert result.is_valid
    assert quote.quality == DataQuality.INDICATIVE
    assert result.issues
