import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Protocol, Optional
from urllib.parse import urlparse

from opensearchpy import OpenSearch


PERFORMANCE_RECORD_LATENCY_MS = "latency"
PERFORMANCE_RECORD_THROUGHPUT_SAMPLE_PER_SEC = "throughput"


@dataclass
class PerformanceRecord:
    metric: str
    kind: str
    value: Any

    when: datetime = field(default_factory=lambda: datetime.now())
    meta: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def latency(metric: str, value_ms: float, meta: Optional[Dict[str, Any]] = None, when: Optional[datetime] = None):
        r"""
        Create a PerformanceRecord tracking latency information

        Args:
            `metric` (`str`):
                Metric identifier
            `value_ms` (`float`):
                The recorded latency, in millisecond, for the underlying metric record
            `meta` (`Optional[Dict[str, Any]]`, defaults to `{}`)
                Information relative to the recorded metric to store alongside the metric readout
            `when` (`Optional[datetime]`, defaults to `datetime.now()`)
                Indicates when the underlying metric was recorded
        Returns:
            The performance record for the target metric representing latency
        """
        return PerformanceRecord(
            metric=metric, kind=PERFORMANCE_RECORD_LATENCY_MS, value=value_ms, when=when, meta=meta
        )

    @staticmethod
    def throughput(metric: str, value_sample_per_sec: float, meta: Optional[Dict[str, Any]] = None, when: Optional[datetime] = None):
        r"""
        Create a PerformanceRecord tracking throughput information

        Args:
            `metric` (`str`):
                Metric identifier
            `value_sample_per_sec` (`float`):
                The recorded throughput, in samples per second, for the underlying metric record
            `meta` (`Optional[Dict[str, Any]]`, defaults to `{}`)
                Information relative to the recorded metric to store alongside the metric readout
            `when` (`Optional[datetime]`, defaults to `datetime.now()`)
                Indicates when the underlying metric was recorded
        Returns:
            The performance record for the target metric representing throughput
        """
        return PerformanceRecord(
            metric=metric,
            kind=PERFORMANCE_RECORD_THROUGHPUT_SAMPLE_PER_SEC,
            value=value_sample_per_sec,
            when=when,
            meta=meta
        )

    def as_document(self) -> Dict[str, Any]:
        r"""
        Convert the actual `PerformanceRecord` to a dictionary based representation compatible with document storage
        Returns:
            Dictionary of strings keys with the information stored in this record
        """
        parcel = { "date": self.when.timestamp(), "metric": self.metric, "kind": self.kind, "value": self.value }
        return parcel | self.meta


class PerformanceTrackerStore(Protocol):

    @staticmethod
    def from_uri(uri: str) -> "PerformanceTrackerStore":
        pass

    def push(self, collection: str, record: "PerformanceRecord"):
        pass



class OpenSearchPerformanceTrackerStore(PerformanceTrackerStore):
    # Extract region and service from AWS url (ex: us-east-1.es.amazonaws.com)
    AWS_URL_RE = re.compile(r"([a-z]+-[a-z]+-[0-9])\.(.*)?\.amazonaws.com")

    def __init__(self, url: str, auth):
        uri = urlparse(url)
        self._client = OpenSearch(
            [{"host": uri.hostname, "port": uri.port or 443}],
            http_auth = auth,
            http_compress = True,
            use_ssl = True
        )

        # Sanity check
        self._client.info()

    @staticmethod
    def from_uri(uri: str) -> "PerformanceTrackerStore":
        if not (_uri := urlparse(uri)).scheme.startswith("es"):
            raise ValueError(f"Invalid URI {uri}: should start with os:// or os+aws://")

        if _uri.scheme == "es+aws":
            from boto3 import Session as AwsSession
            from botocore.credentials import Credentials as AwsCredentials
            from opensearchpy import AWSV4SignerAuth, Urllib3AWSV4SignerAuth

            # Create AWS session from the (eventual) creds
            if not _uri.username and not _uri.password:
                session = AwsSession()
                creds = session.get_credentials()
            else:
                creds = AwsCredentials(_uri.username, _uri.password)

            # Parse the url to extract region and service
            if len(match := re.findall(OpenSearchPerformanceTrackerStore.AWS_URL_RE, _uri.netloc)) != 1:
                raise ValueError(f"Failed to parse AWS es service URL {uri}")

            region, service = match[0]
            auth = Urllib3AWSV4SignerAuth(creds, region, service)
        else:
            auth = (_uri.username, _uri.password)

        return OpenSearchPerformanceTrackerStore(uri, auth)

    def _ensure_collection_exists(self, collection: str):
        if not self._client.indices.exists(collection):
            self._client.indices.create(collection)

    def push(self, collection: str, record: "PerformanceRecord"):
        self._ensure_collection_exists(collection)
        self._client.index(collection, record.as_document())


class AutoPerformanceTracker:

    @staticmethod
    def from_uri(uri: str) -> "PerformanceTrackerStore":
        if uri.startswith("es://") or uri.startswith("es+aws://"):
            return OpenSearchPerformanceTrackerStore.from_uri(uri)

        raise ValueError(
            f"Unable to determine the service associated with URI: {uri}. "
            "Valid schemas are es:// or es+aws://"
        )



