## 2024-03-24 - Reflex Select Accessibility
**Learning:** Reflex `rx.select` components often lack programmatic association with their visual labels, especially when `rx.text` is used separately.
**Action:** Always ensure `rx.select` has an `aria_label` attribute or is properly associated with a label using `html_for` (or similar) or wrapped in a labeled control.
