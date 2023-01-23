import * as tf from '@tensorflow/tfjs';
import * as ui from './ui';


export async function urlExists(url) {
  ui.status('Testing url ' + url);
  try {
    const response = await fetch(url, {method: 'HEAD'});
    return response.ok;
  } catch (err) {
    return false;
  }
}

/**
 * Load metadata file stored at a remote URL.
 *
 * @return An object containing metadata as key-value pairs.
 */
export async function loadHostedMetadata(url) {
  ui.status('Loading metadata from ' + url);
  try {
    const metadataJson = await fetch(url);
    const metadata = await metadataJson.json();
    ui.status('Done loading metadata.');
    return metadata;
  } catch (err) {
    console.error(err);
    ui.status('Loading metadata failed.');
  }
}
