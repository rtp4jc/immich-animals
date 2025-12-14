# Phase 5: Minimal Immich UI Integration (Revised)

## 1. Objective

To provide a minimal, user-facing integration of the animal identification pipeline within the Immich web UI. This phase focuses on leveraging the existing "People" UI/UX patterns to create a familiar "Pets" section, enabling users to view, name, and manage automatically generated pet clusters.

## 2. Guiding Principles

*   **Reuse, Don't Reinvent**: Aggressively reuse existing components, services, and UI patterns from the people identification feature to minimize development time and maintain a consistent user experience.
*   **Minimalism**: Implement only the core features required for a functional user workflow. Advanced features are out of scope.
*   **Configuration over Code**: The entire feature should be behind a feature flag and an admin-configurable setting.

## 3. Technical Implementation Plan

### 3.1. Step 1: Backend Scaffolding (Immich Server)

1.  **Feature Flag**: In `server/src/services/system-config.service.ts`, add a new feature flag `animalDetection` to the `FeatureFlags` class.
2.  **Database Entity**: Create `server/src/repositories/entities/pet.entity.ts` that mirrors `server/src/repositories/entities/person.entity.ts`.
3.  **Repository**: Create `server/src/repositories/pet.repository.ts` that mirrors `server/src/repositories/person.repository.ts`.
4.  **Service Layer**: Create `server/src/services/pet.service.ts` that mirrors `server/src/services/person.service.ts`.
5.  **Controller**: Create `server/src/controllers/pet.controller.ts` that mirrors `server/src/controllers/person.controller.ts`.
6.  **Module**: Update `server/src/app.module.ts` to include the new `PetService` and `PetRepository`.

### 3.2. Step 2: Job & Machine Learning Integration (Immich Server)

1.  **Job Scheduling**: In `server/src/services/job.service.ts`, create a new job `animal-detection` to be queued when the `animalDetection` feature flag is enabled.
2.  **Machine Learning Communication**: In the `JobService`, when processing the `animal-detection` job, the server will call the `/predict` endpoint of the `immich-machine-learning` container.

    *   **Endpoint**: `http://<machine-learning-host>:<port>/predict`
    *   **Method**: `POST`
    *   **Payload**: The request will be a `multipart/form-data` request containing the image file and a JSON payload with the following structure:

        ```json
        {
            "entries": {
                "dog-identification": {
                    "detection": {"modelName": "dog_detector"},
                    "recognition": {"modelName": "dog_embedder_direct"}
                }
            }
        }
        ```

    *   **Response**: The server will expect a JSON response containing an array of embeddings, which will then be stored in the database.

### 3.3. Step 3: Frontend Implementation (Immich Web)

1.  **Create "Pets" Page**:
    *   Create a new directory `web/src/routes/(user)/pets`.
    *   Inside, create `+page.svelte` and `+page.ts` files that mirror `web/src/routes/(user)/people/+page.svelte` and `+page.ts`.
    *   The `+page.svelte` will reuse the `people-infinite-scroll.svelte` component, fetching data from the new `/api/pets` endpoint.
2.  **Create "Pet" Page**:
    *   Create a new directory `web/src/routes/(user)/pets/[petId]`.
    *   Inside, create `+page.svelte` and `+page.ts` files that mirror `web/src/routes/(user)/people/[personId]/+page.svelte` and `+page.ts`.
3.  **Update Navigation**: In `web/src/lib/components/layouts/app-layout.svelte`, add a new navigation item for "Pets" that links to `/pets`.
4.  **Implement Cluster Management**:
    *   Reuse components from `web/src/lib/components/faces-page/` (e.g., `merge-face-selector.svelte`) with minor modifications to work with the "Pets" data and API endpoints.
5.  **Add Settings Toggle**: In `web/src/routes/admin/settings/+page.svelte`, add a toggle for the `animalDetection` feature flag under the "Recognition" section.

## 4. Implementation Hints

*   **Backend Scaffolding**: Use the NestJS CLI to generate the new `pet` module, service, and controller (`nest generate module pet`, `nest generate service pet`, `nest generate controller pet`). This will ensure consistency with the existing project structure.
*   **API Client**: The Immich web app uses a generated OpenAPI client. After adding the new `pet` controller and its endpoints, regenerate the client by running the appropriate command (likely a `pnpm` script in the `web` directory). This will make the new endpoints available in the frontend without manual fetching.
*   **Permissions**: The `Person` entity has relationships with `User` for ownership and permissions. The new `Pet` entity should have a similar relationship to ensure that only authorized users can view and manage pet information.
*   **Error Handling**: When communicating with the machine learning service, implement robust error handling to gracefully handle cases where the service is unavailable or returns an error.
*   **Testing**: Add unit tests for the new `PetService` and integration tests for the `PetController` to ensure the new endpoints work as expected.

## 5. Out of Scope for Phase 5

*   **Mobile UI**: This phase focuses on the web UI only.
*   **On-device fine-tuning/personalization**.
*   **Advanced search** (e.g., by breed).
*   **Active learning**.
*   **Pet-specific UI elements** (e.g., custom icons).

## 6. Success Criteria

*   Users can enable the pet detection feature in settings.
*   The system processes assets to find and cluster pets.
*   Users can view, name, merge, and split pet clusters on the "Pets" page.
*   The feature is stable and does not negatively impact application performance.