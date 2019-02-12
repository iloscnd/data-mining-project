# Data mining project

Projekt z przedmiotu eksploracja danych

## Raport

Raport znajduje się w pliku report.ipynb (jupyter notebook)

## Kod

Aby uruchomić klasyfikator bazujący na mierze inbalance użyj komendy 
`python3 imbalance.py sciezka_do_danych `

Aby uruchomić trenowanie sieci neuronej użyj komendy
`python3 main.py --data sciezka_do_danych`
Wszystkie mozliwe komendy dostępne są po wpisaniu polecenia:
`python3 main.py -h`
Dołączone modele na których bazuje raport uruchomione są za pomocą komendy:
`python main.py --data sciezka_do_danych --debug --epochs 300 --bucket_size 0.04 --buckets 3 --hidden 1000`
