# 1. Розгортання проекту

```bash
git clone git@github.com:arlien3560/redko_nosql_2.git
```

```bash
cd redko_nosql_2
```

```bash
python3 -m venv venv
```

```bash
source venv/bin/activate
```

```bash
pip install -r requirements.txt
```

```bash
python main.py
```

Директорія data спеціально занесена в .gitignore через обсяг файлу - для збереження в репозиторії файл має складати не більше 100МБ.
Також в ході роботи доставила додаткові залежності, інкаше виникає помилка запису файлу parquet. Підправила шляхи для збереження даних.


# 2. Task 1.2 

1. Вибір інструментів - нижче наведу порівняльну таблицю для наглядності


| Характеристика     | Pinecone             | Qdrant                    | Chroma            |
|--------------------|----------------------|---------------------------|-------------------|
| **Розгортання**    | Тільки managed SaaS  | Self-host, cloud, hybrid  | Embedded + server |
| **Ліцензія**       | Пропрієтарна         | Apache 2.0                | Apache 2.0        |
| **Масштаб**        | Авто, до сервісу     | До мільярдів векторів     | Мільйони векторів |
| **Продуктивність** | Стабільна            | Сильна у filtered search  | Слабша на write   |
| **Найкраще для**   | Production, zero-ops | Performance/$, фільтрація | Прототипування    |

**Chroma** - оберу для пет-проекту і для одноразових підрахунків.
**Pinecone** - оберу для більш-менш серйозного проекту, але якщо хочу зекономити на DevOps підтримці
**Qdrant** - оберу для високонавантаженого проекту з високими вимогами до швидкості запиту і конкурентними запитами

2. По суті ми підключаємо чисто модель і очікуємо, що будемо працювати саме з науковими статтями (модель навчалась на 6M триплетів наукових цитувань, кодує title + abstract як єдине ціле). Обраний specter2_base спеціалізується саме на научних більш масивних текстах і має більше параметрів, а all-MiniLM-L6-v2 є все ж таки базовою моделлю. Specter2_base займе більше місця в Pinecone судячи по характеристикам. Також має більший вхід - вдвічі більший з порівнюваноою all-MiniLM-L6-v2. Менше помилятись будуть. В самій документації також описано, що вона якраз і призначена для Proximity сценарію з коробки.

Цитата
"SPECTER2 is the successor to SPECTER and is capable of generating task specific embeddings for scientific tasks when paired with adapters. This is the base model to be used along with the adapters. Given the combination of title and abstract of a scientific paper or a short texual query, the model can be used to generate effective embeddings to be used in downstream applications." - https://huggingface.co/allenai/specter2_base

3. Насправді про метрику безпосередньо в картці нічого не вказано - були тільки фрагменти того як модель навчалась. Побачивши, що вказаний формат даних '{"query", "pos", "neg"}', а також звернувши увагу на допис "The citation link are triplets in the form", і подумала про "triplet margin loss". А вже нижче майже відразу була стаття за посиланням про model training яка чітко вказує, що використовувалась L2 відстань для тренування, тобто Евклідова відстань.
З практичної точки зору необхідно обирати ту метрику, на якій модель навчалась, бо це дасть найбільш точний і коректний результат.


# 3. Task 1.3

1. Додала скрін в директорію зі сткрінами - іллюстрацію запуску.

2. Відповідь на питання:
**Косинусна схожість = (скалярний добуток) / (добуток норм)**
Якщо подивитись на формулу косинусної схожості, то в її знаменнику використовуються норми векторів.
А оскільки це 1, то в знаменнику ми отримуємо добуток двох одиниць. Для обрахунків значущою частною є тільки чисельник. А це - скалярний добуток (dot product). За рахунок того, що ми не рахуємо косинуси, ми таким чином спрощуємо формулу обчислення. А це призведе до пришвидшення пошуку.

# 4. Task 2

Додала комментарі щодо кроків виконання напряму в файл. Скріни результатів додані в папку screens.

# 5. Task 3

1. Моя видача щодо виконання п4 задачі

```javascript
Результати пошуку з фільтрацією (Приклад A):
Результати пошуку з фільтрацією (Приклад B):
Результат 1:
  Назва: Capturing knots in polymers
  Категорія: cond-mat.soft
  Рік: 2007
  Абстракт: This paper visualizes a knot reduction algorithm...

Результат 2:
  Назва: Symbolic sensors : one solution to the numerical-symbolic interface
  Категорія: physics.ins-det
  Рік: 2007
  Абстракт: This paper introduces the concept of symbolic sensor as an extension of the
smart sensor one. Then, ...

Результат 3:
  Назва: The Mathematics
  Категорія: math.HO
  Рік: 2007
  Абстракт: This is an essay that considering the knowledge structure and language of a
different nature, attemp...

Результат 4:
  Назва: Modeling the field of laser welding melt pool by RBFNN
  Категорія: physics.comp-ph
  Рік: 2007
  Абстракт: Efficient control of a laser welding process requires the reliable prediction
of process behavior. A...

Результат 5:
  Назва: Why should anyone care about computing with anyons?
  Категорія: quant-ph
  Рік: 2007
  Абстракт: In this article we present a pedagogical introduction of the main ideas and
recent advances in the a... ```

З цього результату роблю висновок, що за заданими критеріями в варіанті А просто не було статей. 
Всі статті з загаданою темою в моєму випадку були видані в 2007 році, вибірка це підсвічує.
```

2. Виконання п5 задачі. Видача

```javascript
Норми ембеддингів: min=1.000000, max=1.000000, mean=1.000000
Топ-5 статей за cosine similarity:
  Cosine: 0.8294
  Назва: Capturing knots in polymers
  Категорія: cond-mat.soft
  Рік: 2007
  Абстракт: This paper visualizes a knot reduction algorithm...

  Cosine: 0.8260
  Назва: Symbolic sensors : one solution to the numerical-symbolic interface
  Категорія: physics.ins-det
  Рік: 2007
  Абстракт: This paper introduces the concept of symbolic sensor as an extension of the
smart sensor one. Then, ...

  Cosine: 0.8254
  Назва: The Mathematics
  Категорія: math.HO
  Рік: 2007
  Абстракт: This is an essay that considering the knowledge structure and language of a
different nature, attemp...

  Cosine: 0.8181
  Назва: Modeling the field of laser welding melt pool by RBFNN
  Категорія: physics.comp-ph
  Рік: 2007
  Абстракт: Efficient control of a laser welding process requires the reliable prediction
of process behavior. A...

  Cosine: 0.8142
  Назва: Python for Education: Computational Methods for Nonlinear Systems
  Категорія: nlin.CD
  Рік: 2007
  Абстракт: We describe a novel, interdisciplinary, computational methods course that
uses Python and associated...

Топ-5 статей за dot product:
  Dot Product: 0.8294
  Назва: Capturing knots in polymers
  Категорія: cond-mat.soft
  Рік: 2007
  Абстракт: This paper visualizes a knot reduction algorithm...

  Dot Product: 0.8260
  Назва: Symbolic sensors : one solution to the numerical-symbolic interface
  Категорія: physics.ins-det
  Рік: 2007
  Абстракт: This paper introduces the concept of symbolic sensor as an extension of the
smart sensor one. Then, ...

  Dot Product: 0.8254
  Назва: The Mathematics
  Категорія: math.HO
  Рік: 2007
  Абстракт: This is an essay that considering the knowledge structure and language of a
different nature, attemp...

  Dot Product: 0.8181
  Назва: Modeling the field of laser welding melt pool by RBFNN
  Категорія: physics.comp-ph
  Рік: 2007
  Абстракт: Efficient control of a laser welding process requires the reliable prediction
of process behavior. A...

  Dot Product: 0.8142
  Назва: Python for Education: Computational Methods for Nonlinear Systems
  Категорія: nlin.CD
  Рік: 2007
  Абстракт: We describe a novel, interdisciplinary, computational methods course that
uses Python and associated...

Топ-5 статей за L2-distance:
  L2 Distance: 0.5842
  Назва: Capturing knots in polymers
  Категорія: cond-mat.soft
  Рік: 2007
  Абстракт: This paper visualizes a knot reduction algorithm...

  L2 Distance: 0.5899
  Назва: Symbolic sensors : one solution to the numerical-symbolic interface
  Категорія: physics.ins-det
  Рік: 2007
  Абстракт: This paper introduces the concept of symbolic sensor as an extension of the
smart sensor one. Then, ...

  L2 Distance: 0.5910
  Назва: The Mathematics
  Категорія: math.HO
  Рік: 2007
  Абстракт: This is an essay that considering the knowledge structure and language of a
different nature, attemp...

  L2 Distance: 0.6032
  Назва: Modeling the field of laser welding melt pool by RBFNN
  Категорія: physics.comp-ph
  Рік: 2007
  Абстракт: Efficient control of a laser welding process requires the reliable prediction
of process behavior. A...

  L2 Distance: 0.6095
  Назва: Python for Education: Computational Methods for Nonlinear Systems
  Категорія: nlin.CD
  Рік: 2007
  Абстракт: We describe a novel, interdisciplinary, computational methods course that
uses Python and associated...
```

3. Чи збігаються топ-5 для cosine і dot product і чому?
Збігаються. Як зазначала в минулих замітках - при нормалізації для косинусної схожості значущим залишається лише скалярний добуток в чисельнику, тому використала в формулі напряму dot.

4. Чи відрізняються результати для L2 і чому?
Додала в вивід для наглядності відстані.
L2 менше значення означає більшу схожість, в cosine/dot навпаки. Тому в L2 ми беремо найменші значення.
Якщо подивитись на вибірку статей, то не відрізняються по значенням виводу - це одні і ті самі статті. Тобто по суті змінюється лише шкала чисел.

5. Що сталося б, якби ембеддинги не були нормалізовані?
Cosine би сходилась, бо в формулі закладено нормалізацію (знаменник).
Dot давав би інший результат через довжину вектора - менш релевентні піднялись би нагору в видачі через довжину вектора.
L2-distance - теж чутлива до довжини, також змінить результат.
Тож схоже, що саме нормалізація вплинула на результьти. А за рахунок того, що вектори нормалізовані, то ми ще і на обчисленнях зекономили.


# 6. Task 4

Відповіді на питання

1. Стратегія осмислених семантичних чанків більше зберігає контекст (не обривається за сенсом на відміну від Fixed-size), тому і результат її буде кращим. 
Це буде помітно на великих статтях.

2. Так, у fixed-size chunking розрізані речення є; у semantic chunking - ні.
Ембеддинг розрізаного речення гіршої якості, ніж ембеддинг цілого. Модель кодує текст на основі семантики, а обірвана фраза несе неповний або спотворений зміст.

3. Чим більший overlap, тим менший крок вікна, тому тим більше чанків утворюється з того самого тексту.
Більше чанків означає більше ембеддингів, більше місця в індексі та довший час обробки й пошуку.
Overlap дублює прикордонні слова, тому контекст на межах зберігається в обох сусідніх чанках. Релевантний фрагмент не випадає з пошуку через невдалий розріз.
Водночас, overlap створює надлишковість. Сусідні чанки з однаковими словами дають близькі ембеддинги, і це призводить до задвоєння у видачі. По суті це повнота ціною дублювання результатів.
Як видає пошуковик, в реальних системах ця проблема вирішується дедуплікацією за документом при видачі.


# 7. Task 5

Видача

```javascript
Query: BERT fine-tuning

BM25 Top-5:
  [8389] The NMSSM Solution to the Fine-Tuning Problem, Precision Electroweak
   | BM25: 10.5835
  [3658] Conformal dynamics in gauge theories via non-perturbative
  renormaliz  | BM25: 8.1427
  [2569] Inverse Monte-Carlo determination of effective lattice models for SU(3  | BM25: 7.4195
  [4269] Eternal Inflation is "Expensive"  | BM25: 7.0648
  [4385] String tension and removal of lattice coarsening effects in Monte Carl  | BM25: 6.8876

Vector Search Top-5:
  [6406] Misere quotients for impartial games: Supplementary material  | cos: 0.8645
  [2535] Introduction to Phase Transitions in Random Optimization Problems  | cos: 0.8533
  [6795] Abstract Convexity and Cone-Vexing Abstractions  | cos: 0.8500
  [8935] The Compositions of the Differential Operations and Gateaux Directiona  | cos: 0.8481
  [4441] Experimental local realism tests without fair sampling assumption  | cos: 0.8473

Hybrid Search (RRF) Top-5:
  [8389] The NMSSM Solution to the Fine-Tuning Problem, Precision Electroweak
   | RRF: 0.0164
  [6406] Misere quotients for impartial games: Supplementary material  | RRF: 0.0164
  [3658] Conformal dynamics in gauge theories via non-perturbative
  renormaliz  | RRF: 0.0161
  [2535] Introduction to Phase Transitions in Random Optimization Problems  | RRF: 0.0161
  [2569] Inverse Monte-Carlo determination of effective lattice models for SU(3  | RRF: 0.0159



Query: Yann LeCun convolutional networks

BM25 Top-5:
  [281] On Punctured Pragmatic Space-Time Codes in Block Fading Channel  | BM25: 13.4827
  [1410] Trellis-Coded Quantization Based on Maximum-Hamming-Distance Binary
    | BM25: 13.3659
  [1848] Response of degree-correlated scale-free networks to stimuli  | BM25: 8.4662
  [5549] Numerical evaluation of the upper critical dimension of percolation in  | BM25: 7.8511
  [196] Analysis of random Boolean networks using the average sensitivity  | BM25: 7.6376

Vector Search Top-5:
  [4213] Multilayer Perceptron with Functional Inputs: an Inverse Regression
    | cos: 0.8479
  [4821] The Netsukuku network topology  | cos: 0.8431
  [8935] The Compositions of the Differential Operations and Gateaux Directiona  | cos: 0.8429
  [610] Modeling the field of laser welding melt pool by RBFNN  | cos: 0.8346
  [7372] Adaptive classification of temporal signals in fixed-weights recurrent  | cos: 0.8314

Hybrid Search (RRF) Top-5:
  [1143] Optimization in Gradient Networks  | RRF: 0.0299
  [281] On Punctured Pragmatic Space-Time Codes in Block Fading Channel  | RRF: 0.0164
  [4213] Multilayer Perceptron with Functional Inputs: an Inverse Regression
    | RRF: 0.0164
  [1410] Trellis-Coded Quantization Based on Maximum-Hamming-Distance Binary
    | RRF: 0.0161
  [4821] The Netsukuku network topology  | RRF: 0.0161



Query: making computers understand human emotions from text

BM25 Top-5:
  [3661] An Automated Evaluation Metric for Chinese Text Entry  | BM25: 17.6241
  [7897] Towards Understanding the Origin of Genetic Languages  | BM25: 16.8584
  [3664] On the Development of Text Input Method - Lessons Learned  | BM25: 15.7392
  [7321] Detecting anchoring in financial markets  | BM25: 12.3045
  [8443] Philosophy and Relativity  | BM25: 11.8299

Vector Search Top-5:
  [4893] Opinion Dynamics and Sociophysics  | cos: 0.8287
  [3664] On the Development of Text Input Method - Lessons Learned  | cos: 0.8228
  [5681] Extracting the hierarchical organization of complex systems  | cos: 0.8092
  [1157] Novelty and Collective Attention  | cos: 0.8028
  [2541] Narratives within immersive technologies  | cos: 0.8021

Hybrid Search (RRF) Top-5:
  [3664] On the Development of Text Input Method - Lessons Learned  | RRF: 0.0320
  [3661] An Automated Evaluation Metric for Chinese Text Entry  | RRF: 0.0164
  [4893] Opinion Dynamics and Sociophysics  | RRF: 0.0164
  [7897] Towards Understanding the Origin of Genetic Languages  | RRF: 0.0161
  [5681] Extracting the hierarchical organization of complex systems  | RRF: 0.0159
```

1. Слід зауважити, що видані статті не мають релевантного співпадіння з контекстом запитів, але знайшов віддалено схожі за змістом теми. Найімовірніша причина - такої інформації в базі немає взагалі.
BM25 - вчепився за збіг слів. Побачив термін - зметчив.
Vector - вхоплює дотичну тему, хоч прямих термінів нема.
RRF - взяв від обох систем краще, тож він трохи стабільніший.

На поточних даних неможливо однозначно сказати, який кращий, але більш змістовний результат в теорії має дати якраз останній варіант при наявності релевантних даних.

2. Якщо так подивитись, то можна звернути увагу на Optimization in Gradient Networks в кейсі запиту Yann LeCun convolutional networks
Система винагороджує тільки ті документи, де обидва дочірні методи показують згоду. Тому і штовхають її в видачі нагору.

3. Якщо подивитись на формулу, то вона виглядає так - (1/k+rank)
Якщо підставити k=60, то значущі цифри будуть близькі один до одного при зміні rank, тож отримаємо найбільшу стійкість до випадкових значень
Якщо підставити k=1, то буде сильна чутливість до випадкового топового значення.



# 8. Task 6

1. BM25 порівнює слова, семантичний пошук порівнює зміст.
BM25 сильно зав'язаний на лексиці: чим частіше згадуються в тексті слова, тим вище в видачі документ.
Семантичний пошук кодує запит і документ у вектори через specter2_base, тому за принципом пошуку векторів знаходить тематично близькі тексти тоді, коли спільних слів немає.

Якщо подивитись на мою роботу, то видно, що BM25 виграв на точних термінах: при пошуку "BERT fine-tuning" він підняв статтю The NMSSM Solution to the Fine-Tuning Problem. Хоча це фізика,  точність терміна тут зіграла проти релевантності. На запиті "making computers understand human emotions from text" BM25 вивів статті зі словами "text" в назві "On the Development of Text Input Method".
Семантичний пошук виграв на перефразуванні без явних термінів: на запиті про емоції вектор знайшов Opinion Dynamics and Sociophysics і Narratives within immersive technologies. Вектор уловив дотичну тему людської поведінки та сприйняття.

Висновок
BM25 треба використовувати на точний пошук слів: усе, де користувач знає точну лексику і хоче саме її. 
Семантичний пошук кращий для запитів, сформульованих своїми словами, коли користувач не знає точної термінології документа.
На практиці найкраще працює гібрид (як у завданні 06): BM25 ловить точні збіги, вектор ловить зміст, а RRF поєднує обидва.

2. Якщо чанк занадто маленький (10–15 слів), він втрачає контекст: фрагмент стає надто коротким, щоб нести завершену думку. Ембеддинг виходить беззмістовним. Це я бачила прямо у завданні 5: чанки на кшталт "in the past" потрапляли у видачу, не несучи жодної корисної інформації. Маленькі чанки різко збільшують кількість чанків, роздувають індекс і створюють багато шуму. Релевантна відповідь може виявитися розподіленою між двома чанками так, що жоден з них не міститиме її повністю.
Якщо чанк занадто великий (500+ слів), то виникає розмивання змісту. Більшість sentence-transformer моделей мають максимальну довжину входу в 512 токенів, і текст понад нього просто обрізається. Частина чанка не кодується взагалі. Великий чанк, що містить кілька різних тем, дає ембеддинг, який не представляє добре жодну з тем. Якщо у великому чанку лише одне речення з тридцяти вагоме, його сигнал тоне в загальному векторі, і пошук гірше знаходить цей чанк за вузьким запитом. Точність локалізації падає: знайшовши чанк, користувач отримує велику стіну тексту замість конкретного релевантного фрагмента.
Оптимальний розмір залежить від задачі, єдиного правильного числа немає. Рекомендується діапазон 200–500 слів (або 256-512 токенів) з невеликим overlap. Вибір залежить від вигляду даних і запитів: для пошуку конкретних фактів кращі менші чанки (точна локалізація), для пошуку за загальною темою чи для RAG більші. 
Ключові важелі для рішення: ліміт токенів моделі (верхня межа), структура тексту (краще різати по абзацах, реченнях), і тип запитів (вузькі факти vs широкі теми). Тому краще визначитись для якої задачі розподіляємо чанк, а потім вже підбирати найкращу конфігурацію.

3. Я вже зазначала вище за ходом роботи, різниці в видачі що в випадку cosine, що в випадку l2, не буде. Абсолютні значення скорів при цьому відрізнятись будуть. І це відбувається саме завдяки нормалізації.

Зв'язок виводиться так.

Квадрат L2-відстані в загальному вигляді:
    ||a - b||² = ||a||² + ||b||² - 2·(a·b)

Для нормалізованих векторів ||a||² = ||b||² = 1, тому два перші члени стають одиницями:
    ||a - b||² = 1 + 1 - 2·(a·b) = 2 - 2·(a·b)

А для одиничних векторів скалярний добуток дорівнює косинусу кута між ними (a·b = cos θ), отже:
    ||a - b||² = 2 - 2·cos(θ)

Звідси видно: L2-відстань монотонно спадає зі зростанням cos(θ). Чим більша косинусна схожість, тим менша відстань — тому порядок ранжування за обома метриками однаковий.

Чим більша косинусна схожість (ближче за змістом), тим менша L2-відстань, і зв’язок строго монотонний. 
Ранжування залежить лише від порядку значень, а не від самих значень, порядок за зростанням L2 завжди збігається з порядком за спаданням cosine.
От приклад з завдання 4 - для документа з cosine = 0.8294 формула дає 2−2×0.8294=0.5841\sqrt{2 - 2 \times 0.8294} = 0.5841
2−2×0.8294​=0.5841,  а прямий розрахунок L2 дав 0.5842. По суті різниця тільки в точності останнього знака. Але результат видачі був однаковий.

4. Ну, по-перше все працювало повільніше - система явно гальмує безкоштовних користувачів. Але для навчальних обчислень нормально.
По-друге, якщо б треба було більше індексів, то були б обмеження на безкоштовному плані.
По третє, тут немає спробувати функції масштабування - також в платному плані.
В четверте - якщо б тут було навантаження і більша кількість статей (10 мільйонів), то я б вже брала Qdrant.
Бо це вже інший обсяг: 10M векторів × 768 × 4 байти (float32) = приблизно 30 ГБ лише на вектори. Метадані та індексні структури якусь вагу також будуть мати. Такий прямий підхід як тут в задачі не можна було б використовувати, бо це було б дуже повільно. Потрібен був би ANN, який би знизив точність, але пришвидшив би видачу результату. Ну і звісно, треба було би робити квантизацію. А також потрібно було б робити багатошаровий пошук (фільтр - BM25 - ANN - переранжування).