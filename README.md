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