# Checklist for building code of OTFS detection using DIP

## Checklist of main.py
- [x] Generate transmit symbol with desired size
- [x] Call functions/classes from txrx.py without trouble
- [x] Call functions/classes from tools.py without trouble
- [x] function run: run the DIP algorithm to detect the symbols
- [ ] Debugging

## Checklist of tools.py
- [x] function transmit: generate a modulated symbols
- [x] function receive: return the symbols that has been added by gaussian noise
- [ ] function SER: Compare it with true x
- [ ] Need more functions?
- [ ] Debugging

## Checklist of model.py
- [ ] Define the NN architecture
- [ ] Modify the reference code
- [ ] Debugging
