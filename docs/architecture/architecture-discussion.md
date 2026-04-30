### Backend
* Python makes more sense. NLP ecosystem is written with Python.
* ORM:
    - SQLALchemy, Alembic
    - Alternative: No ORM and native queries. Less overhead, more efficient, but less abstraction and modeling can get more difficult. 
  
* Task Queues:
    - Celery the default option in python.
    - Message brokers:
      - Redis (If we are using Redis for cache, it would be best to also use it as message broker also)
      - RabbitMQ.

### Software architecture:

* Ports-and-adapters is a good option but could get problematic.
* Onion architecture could be an alternative, but it can get complex just like ports and adapters. The layers can be
  dependent from the upper layer, but the domain layer has to be totally independent. Very good for domain-heavy
  applications.
* Both require discipline.

#### A comparison between both architectures:
#### Onion:
```
If you say:

My use case should not depend on SQLAlchemy.
You are thinking in Onion Architecture

Outer:
FastAPI
OpenAI
sentence-transformers
HDBSCAN
PostgreSQL

Application:
ClusterThemesUseCase

Domain:
Document
ThemeCluster
ClusterLabel
SimilarityScore
```

#### Ports-and-adapters

```
If you say:

My use case needs an OrderRepository port, and SQLAlchemy will be the adapter.
You are thinking in Ports & Adapters.

app/
  domain/
  application/
    ports/
      in/
      out/
  adapters/
    in_api/
    in_kafka/
    out_db/
    out_s3/
    out_llm/
  infrastructure/
 ```

Same design, different language.

```
  FastAPI route # inbound adapter
  ↓
  Use case # application layer
  ↓
  Domain object # domain layer
  ↓
  Repository Protocol # outbound port
  ↓
  SQLAlchemy repository # outbound adapter   
 ```

### Frontend
* Vue.js is easier to use than React (At the start).

From ChatGPT:

```text
* Vue is more like a progressive framework. It gives you more built-in conventions around templates, reactivity,
  component structure, styling, and common patterns.
* React mixes markup and logic through JSX:JSX is powerful, but it means you are writing “HTML-like syntax inside
  JavaScript.” That is elegant once you understand it, but it can be mentally heavier at first.
* Vue has equivalents too, but the official ecosystem feels more unified.

Use vue if you:
  * you want faster onboarding
  * you like HTML-like templates
  * you want clearer conventions
  * you prefer less boilerplate
  * you want a balanced framework experience
  * you are building medium-sized apps quickly
    
For scientific apps with many dashboards, I would usually choose React, unless the team strongly prefers Vue.
Not because Vue is bad for this. Vue can absolutely build scientific dashboards. But React has a stronger ecosystem
for complex, research-style visualization apps.

* Typescript vs Javascript. Based on my experience I think typescript is easier to use. From ChatGPT: For a
scientific / visualization-heavy React app, I would choose TypeScript almost every time.

* JavaScript is fine for small prototypes, but once you have clustering data, embeddings, documents, filters,
  chart events, API responses, and shared UI state, TypeScript pays off quickly.
```

### Database:

* How important are relations between tables ?
* NoSQL with no normalization could make sense for rapid prototyping, easier to make changes in the documents. But
  there's more duplication of the information. Mongo could be an option with the community edition: https://www.mongodb.com/products/self-managed/community-edition
* Postgres is good with concurrency.

### Cache:

* If needed Redis makes sense, specially if redis is also the message broker.

### Secrets:

* Hashicorp vault ?

### Logging:
* default logging / structlog
* Metrics → Prometheus client or prometheus-fastapi-instrumentator
* OpenTelemetry
* Observability: Grafana / Prometheus / Loki / Tempo / ELK / Datadog / etc.
* Wandb for model training? 

