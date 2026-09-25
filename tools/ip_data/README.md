# Mainland China CIDR data

The IPv4 and IPv6 CIDR files in this directory are the aggregated China (CN)
exports from [ipverse/country-ip-blocks](https://github.com/ipverse/country-ip-blocks),
retrieved on 2026-08-07. The upstream data is released under CC0-1.0.

The application uses these files only to select the first model download source
from the server's public egress IP. No address or selection result is exposed in
the WebUI or written to its run logs.
