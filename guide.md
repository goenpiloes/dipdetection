# Checklist for building code of OTFS detection using DIP

## Checklist of main.py
- [ ] Generate transmit symbol with desired size
- [ ] Call functions/classes from txrx.py without trouble
- [ ] Call functions/classes from check.py without trouble
- [ ] Debugging

## Checklist of tools.py
- [ ] function transmit: generate a modulated symbols
- [ ] function receive: return the symbols that has been added by gaussian noise
- [ ] function detect: run the DIP algorithm to detect the algorithm
- [ ] function SER: Compare it with true x
- [ ] Need more functions?
- [ ] Debugging

## Checklist of model.py
- [ ] Define the NN architecture
- [ ] Modify the reference code
- [ ] Debugging