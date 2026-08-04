"""Tests for Twilio TwiML builders and call placement."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest_mock

from ghostline.telephony.twilio import (
    CallPlaced,
    TwilioService,
    build_stream_twiml,
    build_voice_twiml,
)


class TestBuildStreamTwiML:
    def test_returns_valid_xml(self) -> None:
        twiml = build_stream_twiml("wss://example.com/twilio")
        root = ET.fromstring(twiml)
        assert root.tag == "Response"

    def test_contains_connect_stream_pause(self) -> None:
        twiml = build_stream_twiml("wss://x.io/twilio")
        root = ET.fromstring(twiml)
        tags = [child.tag for child in root]
        assert "Connect" in tags
        assert "Pause" in tags

    def test_stream_url_aparses_correctly(self) -> None:
        twiml = build_stream_twiml("wss://example.com/twilio")
        root = ET.fromstring(twiml)
        connect = root.find("Connect")
        assert connect is not None
        stream = connect.find("Stream")
        assert stream is not None
        assert stream.get("url") == "wss://example.com/twilio"

    def test_ampersand_in_url_is_escaped(self) -> None:
        twiml = build_stream_twiml("wss://x.io/twilio?campaign=demo&persona=urgent")
        # The raw string should contain &amp; not a bare &
        assert "&amp;" in twiml
        # And the XML should still parse with the correct URL
        root = ET.fromstring(twiml)
        stream = root.find("Connect/Stream")
        assert stream is not None
        assert stream.get("url") == "wss://x.io/twilio?campaign=demo&persona=urgent"

    def test_voice_twiml_is_alias_of_stream_twiml(self) -> None:
        assert build_voice_twiml("wss://x.io/twilio") == build_stream_twiml("wss://x.io/twilio")

    def test_no_say_preamble_by_default(self) -> None:
        twiml = build_stream_twiml("wss://x.io/twilio")
        root = ET.fromstring(twiml)
        say = root.find("Say")
        assert say is None, "Default TwiML should not include <Say> preamble"

    def test_pause_length_is_600(self) -> None:
        twiml = build_stream_twiml("wss://x.io/twilio")
        root = ET.fromstring(twiml)
        pause = root.find("Pause")
        assert pause is not None
        assert pause.get("length") == "600"


class TestTwilioService:
    def test_place_call_returns_call_placed(self, mocker: pytest_mock.MockerFixture) -> None:
        # Mock the twilio Client so we don't hit the network.
        mock_client_cls = mocker.patch("ghostline.telephony.twilio.TwilioClient")
        mock_client = mock_client_cls.return_value
        mock_call = mocker.MagicMock()
        mock_call.sid = "CA_test_sid"
        mock_client.calls.create.return_value = mock_call

        svc = TwilioService("AC" + "a" * 32, "token", "+15551234567")
        result = svc.place_call("+15559876543", "wss://x.io/twilio")

        assert isinstance(result, CallPlaced)
        assert result.sid == "CA_test_sid"
        assert result.to_number == "+15559876543"
        assert result.from_number == "+15551234567"
        mock_client.calls.create.assert_called_once()
        call_kwargs = mock_client.calls.create.call_args.kwargs
        assert call_kwargs["to"] == "+15559876543"
        assert call_kwargs["from_"] == "+15551234567"
        assert "wss://x.io/twilio" in call_kwargs["twiml"]

    def test_constructor_passes_credentials(self, mocker: pytest_mock.MockerFixture) -> None:
        mock_client_cls = mocker.patch("ghostline.telephony.twilio.TwilioClient")
        TwilioService("AC" + "a" * 32, "secret_token", "+15551234567")
        mock_client_cls.assert_called_once_with("AC" + "a" * 32, "secret_token")
