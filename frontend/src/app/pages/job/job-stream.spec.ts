import { addLossPoint, formatDuration, parseLoss } from './job-stream';

describe('job stream helpers', () => {
  it('reads the loss and step from tqdm postfixes in any number format', () => {
    expect(parseLoss('5%| | 100/2000 [00:10, loss=0.0794, step=100]', 7)).toEqual({
      step: 100,
      loss: 0.0794
    });
    expect(parseLoss('loss: 1.5e-3', 7)).toEqual({ step: 7, loss: 0.0015 });
    expect(parseLoss('Loss=-0.25 Step: 3', 0)).toEqual({ step: 3, loss: -0.25 });
  });

  it('takes the step from the tqdm counter when the postfix has none', () => {
    // the backend sends this line before the progress message parsed from it
    expect(parseLoss(' 30%|###| 12/40 [00:10<00:20, loss=0.25, epoch=1/2]', 11)).toEqual({
      step: 12,
      loss: 0.25
    });
    expect(parseLoss('loss=0.25 at 1.5/2 it', 11)).toEqual({ step: 11, loss: 0.25 });
  });

  it('ignores lines without a loss, and words that only contain "loss"', () => {
    expect(parseLoss('Saved LoRA checkpoint', 1)).toBeNull();
    expect(parseLoss('closs=0.5', 1)).toBeNull();
    expect(parseLoss('loss=nan', 1)).toBeNull();
  });

  it('appends points, replaces one for the same step, and drops ones that go back', () => {
    const one = addLossPoint([], { step: 1, loss: 0.5 });
    expect(one).toEqual([{ step: 1, loss: 0.5 }]);
    const replaced = addLossPoint(one, { step: 1, loss: 0.4 });
    expect(replaced).toEqual([{ step: 1, loss: 0.4 }]);
    const two = addLossPoint(replaced, { step: 3, loss: 0.3 });
    expect(two).toHaveLength(2);
    expect(addLossPoint(two, { step: 2, loss: 0.1 })).toBe(two);
  });

  it('formats a duration in words a person reads at a glance', () => {
    expect(formatDuration(30)).toBe('under a minute');
    expect(formatDuration(60)).toBe('1 min');
    expect(formatDuration(59 * 60 + 29)).toBe('59 min');
    expect(formatDuration(2 * 3600 + 5 * 60)).toBe('2 h 05 min');
  });
});
