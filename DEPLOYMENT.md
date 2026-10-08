# Remote MLflow Hosting

The Docker configuration builds a real MLflow tracking database and model registry from the original fitted Ridge model. The model is copied without retraining. Build-time checks compare its predictions with the original test snapshot and verify reloading the registered version.

The hosted copy has its own run ID and version, with a tag linking it to the original local experiment. It includes Ridge metrics, model artifacts, training and validation snapshots, test predictions, and the submission notebook. The full two-model comparison remains in the notebook.

A reverse proxy permits public reading and the explicit MLflow search requests used by the UI. Other mutation requests are blocked. This is an instructor-facing snapshot, not a public writable tracking endpoint. Continue logging experiments locally or to a separately authenticated tracking server.

## Deploy on Render

1. Connect a Render account and choose the public `1127ziyan/day3-assignment` repository.
2. Create a Docker web service on the Free plan, using this repository's Dockerfile. The included `render.yaml` can also configure a Blueprint deployment.
3. Wait for the build and deployment to finish.
4. Open the public service URL, navigate to Models, open `bike_demand_forecast`, and select the hosted version.
5. Verify access in a separate browser session. Copy the actual model-version page URL into `submission_links.json` and the notebook's submission-links block.

No remote model URL has been verified yet. Do not submit a guessed service address or the local `127.0.0.1` address as a deployment link.

Render Free services sleep after inactivity and use ephemeral writable filesystems. The model registry snapshot is baked into the image and restored with each new deployment; browser writes are blocked. A rebuild creates new hosted run identities, so recheck the submitted link after a rebuild. See [Render's current free-service limitations](https://render.com/docs/free).

This deployment hosts the model for inspection and retrieval. It does not expose an inference API. Neither Docker image construction nor local seeding alone establishes that remote deployment succeeded.
