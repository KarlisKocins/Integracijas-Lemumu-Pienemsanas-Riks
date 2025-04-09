# Integrācijas lēmumu pieņemšanas rīka lietošanas ceļvedis

## Ievads

Šis ceļvedis ir paredzēts, lai palīdzētu lietotājiem efektīvi izmantot Integrācijas lēmumu pieņemšanas rīku. Rīks ir izstrādāts, lai palīdzētu organizācijām pieņemt informētus lēmumus par sistēmu integrācijas stratēģijām, izmantojot formālu daudzkritēriju lēmumu pieņemšanas modeli.

## Sistēmas prasības

- Python 3.6 vai jaunāka versija
- Tīmekļa pārlūks (web versijai)
- Instalētas Python bibliotēkas no `requirements.txt` faila

## Instalācija

1. Lejupielādējiet rīka failus no GitHub repozitorija
2. Atveriet komandrindu (Command Prompt) un dodieties uz rīka mapi
3. Instalējiet nepieciešamās bibliotēkas:
   ```bash
   pip install -r requirements.txt
   ```

## Lietošanas instrukcijas

### Tīmekļa versija

1. Palaidiet serveri:
   ```bash
   python app.py
   ```
2. Atveriet pārlūkprogrammu un dodieties uz: http://localhost:5000

### Datora versija

Palaidiet programmu:
```bash
python integration_decision_tool.py
```

## Darba process

### 1. Kritēriju definēšana

1. Ievadiet kritēriju nosaukumus
2. Norādiet kritēriju svarus procentos (kopējai summai jābūt 100%)
3. Izvēlieties, vai kritērijs jāpalielina vai jāsamazina
4. Piemēri kritērijiem:
   - Izmaksas
   - Īstenošanas laiks
   - Saderība ar esošajām sistēmām
   - Riska faktori
   - Tehniskā sarežģītība
   - Uzturēšanas vienkāršība

### 2. Integrācijas iespēju definēšana

1. Ievadiet katras iespējas nosaukumu
2. Pievienojiet detalizētu aprakstu
3. Ieteicams definēt vismaz 2-3 dažādas stratēģijas
4. Piemēri iespējām:
   - Līdzāspastāvēšana
   - Daļēja integrācija
   - Pilnīga integrācija
   - Pakāpeniska integrācija

### 3. Iespēju novērtēšana

1. Novērtējiet katru iespēju pret katru kritēriju
2. Ievadiet skaitliskās vērtības:
   - Izmaksas eiro
   - Laiks mēnešos
   - Vērtējums skalā no 1 līdz 10
3. Rīks automātiski normalizē vērtības

### 4. Rezultātu analīze

1. Apskatiet ieteicamo iespēju un tās pamatojumu
2. Pārskatiet visu iespēju vērtējumus
3. Analizējiet grafisko salīdzinājumu
4. Izveidojiet pārskatu par rezultātiem

## Grafisko elementu interpretācija

- **Radar diagramma**: Parāda, kā katrs variants veicas dažādos kritērijos
- **Svaru diagramma**: Vizualizē kritēriju svarus
- **Salīdzinājuma tabula**: Parāda detalizētu skaitlisko salīdzinājumu

## Padomi efektīvai lietošanai

1. **Kritēriju izvēle**:
   - Izvēlieties svarīgākos kritērijus
   - Pārliecinieties, ka kritēriji ir mērāmi
   - Izvairieties no pārāk liela kritēriju skaita

2. **Svaru piešķiršana**:
   - Svarīgākajiem kritērijiem piešķiriet lielākus svarus
   - Pārliecinieties, ka svaru summa ir 100%

3. **Novērtēšana**:
   - Izmantojiet konsekventu novērtēšanas skalu
   - Dokumentējiet novērtējumu pamatojumu
   - Ņemiet vērā gan kvantitatīvos, gan kvalitatīvos faktorus

## Problēmu novēršana

1. **Servera problēmas**:
   - Pārliecinieties, ka Python ir pareizi instalēts
   - Pārbaudiet, vai visas bibliotēkas ir instalētas
   - Pārliecinieties, ka nepieciešamie porti nav aizņemti

2. **Datu ievades problēmas**:
   - Pārliecinieties, ka ievadāt pareizus datu tipus
   - Pārbaudiet, vai svaru summa ir 100%
   - Pārliecinieties, ka visi lauki ir aizpildīti