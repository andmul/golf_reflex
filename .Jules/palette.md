## 2025-02-24 - Reflex Select ARIA Labels
**Learning:** Reflex `rx.select` (v0.8.26) does not pass the `aria_label` prop to the underlying trigger button. Instead, you must use `custom_attrs={"aria-label": "..."}`.
**Action:** Always use `custom_attrs` for aria-label on `rx.select` components.
