# Historical adapter recovery

Recovery was attempted twice against stopped pod `s7bbri3e3zilb5` and network
volume `sqplkjdfcd`. Runpod rejected both starts before provisioning because the
account balance was too low. No GPU time accrued, no files were transferred,
and the pod remains stopped. The empty recovery manifest is deliberate; it does
not represent a recovered adapter.
