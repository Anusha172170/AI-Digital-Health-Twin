# Device support matrix

No physical devices have been tested or connected in this repository.

| Path | Intended use | Status |
|---|---|---|
| Android Health Connect | Read records the wearable companion app writes, with permission | Planned; needs device/app/OS details |
| Apple HealthKit | Read authorized health records through a native iOS app | Planned; needs native build/signing and device test |
| Fitbit API | Authorized provider-cloud data | Planned; needs app registration and OAuth |
| Oura API | Authorized ring-cloud data | Planned; needs provider access and OAuth |
| WISDM watch feature files | Offline activity benchmark | Implemented; not a live connector |

Availability and cadence depend on the device, companion app, granted permissions and provider policies. BP/glucose are not assumed available from a consumer watch or ring. Reading steps does not imply access to raw accelerometer samples.
