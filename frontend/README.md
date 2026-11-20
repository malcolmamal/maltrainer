# Frontend

The maltrainer web UI: set up a training run, follow it live, and manage the queue. Angular 20, standalone
components and signals, served by `npm start` on http://localhost:3003 with `/api` and `/ws` proxied to the
backend on port 8008 (`run_webui.bat` / `run_webui.sh` start both).

## Design system

Every screen is built from the components in `src/app/ui` and the colour tokens in `src/styles/_tokens.scss`,
which define a dark and a light theme (the rail's theme switch, or the OS setting). The catalogue, the rules and
how to add a component are in [AGENTS.md](AGENTS.md); `src/app/design-system.guard.spec.ts` fails the tests when
a page writes its own colours or fonts, uses a raw control, or restyles a component.

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
