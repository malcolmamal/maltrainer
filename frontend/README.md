# Frontend

This project was generated using [Angular CLI](https://github.com/angular/angular-cli) version 20.3.13.

## Development server

To start a local development server, run:

```bash
ng serve
```

Once the server is running, open your browser and navigate to `http://localhost:4200/`. The application will automatically reload whenever you modify any of the source files.

## Code scaffolding

Angular CLI includes powerful code scaffolding tools. To generate a new component, run:

```bash
ng generate component component-name
```

For a complete list of available schematics (such as `components`, `directives`, or `pipes`), run:

```bash
ng generate --help
```

## Building

To build the project run:

```bash
ng build
```

This will compile your project and store the build artifacts in the `dist/` directory. By default, the production build optimizes your application for performance and speed.

## Testing

Unit tests run on [Jest](https://jestjs.io) (`jest-preset-angular`, jsdom), not Karma. HTTP goes through
`HttpTestingController`; the job page's WebSocket and the trainer page's timers (poll, plan debounce) are faked.

```bash
npm test                # run all specs
npm run test:coverage   # same, with a coverage report and the threshold check
npm run lint            # ng lint: eslint + prettier, TypeScript and templates
```

Coverage is collected from `src/**/*.ts` (not `main.ts`, specs, environments or `.d.ts`). The global thresholds in
`jest.config.js` (`coverageThreshold`) sit just below the measured values, so `test:coverage` fails when coverage
slips. They may only be raised, never lowered: write the missing test instead. CI runs lint, `test:coverage` and
the build.

## Running end-to-end tests

For end-to-end (e2e) testing, run:

```bash
ng e2e
```

Angular CLI does not come with an end-to-end testing framework by default. You can choose one that suits your needs.

## Additional Resources

For more information on using the Angular CLI, including detailed command references, visit the [Angular CLI Overview and Command Reference](https://angular.dev/tools/cli) page.
