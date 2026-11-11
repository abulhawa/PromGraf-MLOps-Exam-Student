.PHONY: all stop evaluation fire-alert recover

all:
	docker compose up -d --build bike-api prometheus grafana node-exporter

stop:
	docker compose down

evaluation:
	docker compose run --rm --build evaluation

# Stop the API to trigger BikeApiDown after the 1-minute pending period.
fire-alert:
	docker compose stop bike-api

recover:
	docker compose up -d bike-api
